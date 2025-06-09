from .base import *
class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        if self.distance_type == 'l1':
            return torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
        elif self.distance_type == 'csm':
            labels = F.normalize(labels.view(labels.shape[0],-1), dim=1)
            return torch.matmul(labels, labels.T)
        else:
            raise ValueError(self.distance_type)



class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        if self.similarity_type == 'l2':
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        elif self.similarity_type == 'csm':
            features = F.normalize(features, dim=1)
            return torch.matmul(features, features.T)
        else:
            raise ValueError(self.similarity_type)

class ConDTGLoss(nn.Module):
    def __init__(self, device, t = 0.07):
        super(ConDTGLoss, self).__init__()
        self.sim_fn = FeatureSimilarity('csm')
        self.t = t
        self.device = device
    def forward(self, zizs):
        _device = self.device
        zizs_logits = self.sim_fn(zizs).div(self.t)
        zizs_logits_max, _ = torch.max(zizs_logits, dim=1, keepdim=True)
        zizs_logits -= zizs_logits_max.detach()
        zizs_exp_logits = zizs_logits.exp()
        size = len(zizs) // 2
        label = torch.tensor([1] * size + [0] * size).view(-1, 1)
        domain_mask = torch.eq(label, label.T).float().to(_device)
        logits_mask = torch.ones((2*size, 2*size))
        logits_mask.fill_diagonal_(0)
        logits_mask = logits_mask.float().to(_device)
        domain_mask = domain_mask * logits_mask
        log_prob = zizs_logits - torch.log(zizs_exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = domain_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = -(domain_mask * log_prob).sum(1) / mask_pos_pairs

        return mean_log_prob_pos.mean()

class ContrastiveLoss(nn.Module):
    # Modified from Supervised Contrastive Learning Implementation
    # https://github.com/XG293/SupConLoss
    def __init__(self, zizs_temperature, zi_temperature, label_diff, zi_sim, zi_zs_sim):
        super(ContrastiveLoss, self).__init__()
        self.zizs_t = zizs_temperature
        self.zi_t = zi_temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.zi_zs_sim_fn = FeatureSimilarity(zi_zs_sim)
        self.zi_sim_fn = FeatureSimilarity(zi_sim)

        self.denorm_zero = 0

    def forward(self, zi, zs, labels, domains):
        _device = zi.device
        label_diffs = self.label_diff_fn(labels)
        zizs = torch.cat([zi, zs], dim=0)
        zizs_logits = self.zi_zs_sim_fn(zizs).div(self.zizs_t)
        zizs_logits_max, _ = torch.max(zizs_logits, dim=1, keepdim=True)
        zizs_logits -= zizs_logits_max.detach()
        zizs_exp_logits = zizs_logits.exp()
        zi_logits = self.zi_sim_fn(zi).div(self.zi_t)
        zi_logits_max, _ = torch.max(zi_logits, dim=1, keepdim=True)
        zi_logits -= zi_logits_max.detach()
        zi_exp_logits = zi_logits.exp()
        n = zi.shape[0]
        zi_logits = zi_logits.masked_select((1 - torch.eye(n).to(_device)).bool()).view(n, n - 1)
        zi_exp_logits = zi_exp_logits.masked_select((1 - torch.eye(n).to(_device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(_device)).bool()).view(n, n - 1)
        loss = 0.
        for k in range(n - 1):
            pos_logits = zi_logits[:, k]
            pos_label_diffs = label_diffs[:, k]
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()
            pos_log_probs = pos_logits - torch.log((neg_mask * zi_exp_logits).sum(dim=-1))
            loss += - (pos_log_probs / (n * (n - 1))).sum()
        domains = domains.view(-1, 1)
        domain_mask = torch.eq(domains, domains.T).float().to(_device)
        logits_mask = torch.ones((2*n, 2*n))
        logits_mask.fill_diagonal_(0)
        logits_mask = logits_mask.float().to(_device)
        domain_mask = domain_mask * logits_mask
        zizs_exp_logits = zizs_exp_logits * logits_mask
        log_prob = zizs_logits - torch.log(zizs_exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = domain_mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = -(domain_mask * log_prob).sum(1) / mask_pos_pairs
        loss += mean_log_prob_pos.mean()
        return loss

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = args.device
        self.w_oa = args['w_oa']
        self.w_pred = args['w_pred']
        self.w_dtg = args['w_dtg']
        self.encoder,self.di_encoder,self.ds_encoder,self.head,self.fc,self.DTG_head = model_arch[args['dataset']]

        self.conLoss = ContrastiveLoss(
                zizs_temperature=args['zizs_temperature'], 
                zi_temperature=args['zi_temperature'], 
                label_diff=args['label_diff'], 
                zi_sim=args['zi_sim'], 
                zi_zs_sim=args['zi_zs_sim']
        )

        self.conDTGLoss = ConDTGLoss(args.device, args['zizs_temperature'])

    def forward(self, x):
        if self.args.use_batch_norm:
            x, min_vals, max_vals = batch_minmax_norm(x)

        x = self.encoder(x)
        di_enc = self.di_encoder(x).view(x.size(0), -1)
        ds_enc = self.ds_encoder(x).view(x.size(0), -1)

        if self.args.use_batch_norm:
            encode = torch.cat([di_enc, min_vals, max_vals], dim=1)
        else:
            encode = di_enc

        out = self.fc(encode)
        return out.view(-1), (di_enc, ds_enc)


    def loss_function(self, x, y, d):
        out, (di_enc, ds_enc) = self.forward(x)
        prediction_loss = F.mse_loss(out, y)

        di_enc_label_val = self.args.n_domains
        di_enc_labels = torch.full((di_enc.size(0),), di_enc_label_val, dtype=torch.long, device=di_enc.device)
        con_label = torch.cat([di_enc_labels, d], dim=0)

        supCon_loss = self.conLoss(self.head(di_enc), self.head(ds_enc), y, con_label)

        zi_enc = di_enc.detach()
        zs_enc = ds_enc.detach()
        zi_enc_grl = GradReverse.apply(di_enc)
        zizs = torch.concat([zi_enc_grl, zs_enc], dim = 1)
        zizs = self.DTG_head(zizs)
        perm_index = torch.randperm(zs_enc.size(0))
        zi_zs = torch.concat([zi_enc, zs_enc[perm_index]], dim = 1)
        zi_zs = self.DTG_head(zi_zs)
        zizs = torch.concat([zizs, zi_zs], dim = 0)

        disentangle_loss = self.conDTGLoss(zizs)

        return  self.w_pred * prediction_loss + self.w_oa * supCon_loss + self.w_dtg * disentangle_loss, (prediction_loss, supCon_loss)

    def classifier(self, x, y):
        with torch.no_grad():
            if self.args.use_batch_norm:
                x, min_vals, max_vals = batch_minmax_norm(x)
            x = self.encoder(x)
            di_enc = self.di_encoder(x).view(x.size(0), -1)

            encode = di_enc
            if self.args.use_batch_norm:
                encode = torch.cat([di_enc, min_vals, max_vals], dim=1)

            pred = self.fc(encode).view(-1)
            if self.args.unnorm:
                pred = torch.relu(pred * self.args.std + self.args.mean)
                y = y * self.args.std + self.args.mean
        
        return pred, y
    
    def classifier_tsne(self, x, y):
        with torch.no_grad():
            if self.args.use_batch_norm:
                x, min_vals, max_vals = batch_minmax_norm(x)
            x = self.encoder(x)
            di_enc = self.di_encoder(x).view(x.size(0), -1)
            ds_enc = self.ds_encoder(x).view(x.size(0), -1)

            encode = di_enc
            if self.args.use_batch_norm:
                encode = torch.cat([di_enc, min_vals, max_vals], dim=1)

            out = self.fc(encode)
            di_enc = self.head(di_enc)
            ds_enc = self.head(ds_enc)
        return out.view(-1), y, (di_enc, ds_enc)


def res_plot_tsne(source_loaders, target_loader, model, classifier_tsne, idx2name, rep, top_batchs = None):
    model.eval()
    zi_reps = []
    true_y = []
    pred_y = []
    label_T_vs_per_S = []
    label_T_vs_all_S = []
    zi_zs_reps = []
    zi_zs_label = []

    with torch.no_grad():
        for idx, loader in enumerate(source_loaders):
            sample_batch = 0
            for (xs, ys, ds) in loader:
                xs, ys, ds = xs, ys, ds
                sample_batch += 1
                pred, y, (di_enc,ds_enc) = classifier_tsne(xs, ys)
                zi_reps += di_enc.detach().cpu().tolist()
                label_T_vs_per_S += ds.detach().cpu().tolist()
                label_T_vs_all_S += [0] * len(xs)
                true_y += y.detach().cpu().tolist()
                pred_y += pred.detach().cpu().tolist()

                zi_zs_reps += di_enc.detach().cpu().tolist() + ds_enc.detach().cpu().tolist()
                zi_zs_label += [0] * len(xs) + (ds + 3).detach().cpu().tolist()
                if top_batchs is not None and sample_batch > top_batchs:
                    break

        for idx, loader in enumerate(target_loader):
            sample_batch = 0
            for (xs, ys, ds) in loader:
                xs, ys, ds = xs, ys, ds
                sample_batch += 1
                pred, y, (di_enc,ds_enc) = classifier_tsne(xs, ys)
                zi_reps += di_enc.detach().cpu().tolist()
                label_T_vs_per_S += [len(idx2name)-1] * len(xs)
                label_T_vs_all_S += [1] * len(xs)
                true_y += y.detach().cpu().tolist()
                pred_y += pred.detach().cpu().tolist()
                if top_batchs is not None and sample_batch > top_batchs:
                    break
    zi_reps = np.array(zi_reps)
    zi_zs_reps = np.array(zi_zs_reps)
    label_T_vs_per_S = np.array(label_T_vs_per_S)
    label_T_vs_all_S = np.array(label_T_vs_all_S)
    true_y = np.array(true_y)
    pred_y = np.array(pred_y)
    zi_zs_label = np.array(zi_zs_label)
    loc = t_sne(zi_reps)
    zi_zs_loc = t_sne(zi_zs_reps)

    t_sne_plot(loc, label_T_vs_all_S, ['source', 'target'],  f't_SNE_TvsAllS_{rep}.pdf', cmap = "rainbow")
    t_sne_plot_y(loc, true_y,  f't_SNE_true_y_{rep}.pdf', colarbar = False)
    t_sne_plot_y(zi_zs_loc, zi_zs_label,  f't_SNE_zizs_{rep}.pdf', colarbar = False)

def train_one_epoch(args, source_loaders, DEVICE, model, optimizer):
    model.train()
    train_loss = 0
    pred_loss = 0
    con_loss = 0 
    data_size = 0
    for source_loader in source_loaders:
        for x, y, d in source_loader:
            optimizer.zero_grad()

            loss_origin, (prediction_loss, supCon_loss) = model.loss_function(x, y, d)
            loss_origin.backward()
            optimizer.step()

            train_loss += loss_origin * len(d)
            pred_loss += prediction_loss * len(d)
            con_loss += supCon_loss * len(d)
            data_size += len(d)
    train_loss /= data_size
    pred_loss /= data_size
    con_loss /= data_size
    return train_loss, (prediction_loss, con_loss)


def get_accuracy(source_loaders, DEVICE, model, classifier_fn, batch_size, args):
    model.eval()
    y_preds, y_tures = [], []
    metrics = {}
    with torch.no_grad():
        for source_loader in source_loaders:
            for (xs, ys, ds) in source_loader:
                pred_y, ture_y  = classifier_fn(xs, ys)
                pred_y = pred_y.detach().cpu().view(-1) 
                ture_y = ture_y.detach().cpu().view(-1) 
                y_preds += pred_y.tolist()
                y_tures += ys.tolist()

        rmse = mean_squared_error(y_preds, y_tures) ** 0.5
        mae = mean_absolute_error(y_preds, y_tures)
        metrics['mse'] = rmse
        metrics['mae'] = mae

        return metrics


def train(model, DEVICE, optimizer, source_loaders, target_loader, args):
    metrics_recorder = Metrics_tracker(args.metric_list, 'mae', True)
    for e in range(args.n_epoch):
        avg_epoch_loss, (prediction_loss, cont_loss) = train_one_epoch(args, source_loaders, DEVICE, model, optimizer)
        adjust_learning_rate(args.lr, args.lr_decay, e, args.n_epoch, optimizer)
        train_metrics = get_accuracy(source_loaders, DEVICE, model, model.classifier, args.batch_size, args)
        test_metrics = get_accuracy([target_loader], DEVICE, model, model.classifier, args.batch_size, args)
        metrics_recorder.update(train_metrics, test_metrics)
        best_eval_iter = metrics_recorder.get_best_eval_iter('mae')
        best_eval_mse = metrics_recorder.get_test_metric('mse', best_eval_iter)
        best_eval_mae = metrics_recorder.get_test_metric('mae', best_eval_iter)
        tqdm.tqdm.write('Epoch:[{}], avg loss: {:.4f}, y loss: {:.4f}, contrast loss: {:.4f}, train mse:{:.4f} mae:{:.4f}, test mse:{:.4f} mae:{:.4f}' \
                        .format(e + 1, avg_epoch_loss, prediction_loss, cont_loss, train_metrics['mse'], train_metrics['mae'], best_eval_mse, best_eval_mae))
    return best_eval_mae, best_eval_mse
