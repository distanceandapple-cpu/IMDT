import numpy as np
import pandas as pd
import os
import scipy.stats as st
from sklearn.model_selection import KFold
import scanpy as sc
import warnings
import torch
import argparse

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from model.imdt_model import DiT_IMDt
from model.imdt_scheduler import NoiseScheduler
from model.imdt_train import normal_train_IMDt
from model.sample import sample_IMDt
from process.data import *


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='IMD_T')
parser.add_argument("--sc_data", type=str, default="dataset5_seq_915.h5ad")
parser.add_argument("--sp_data", type=str, default='dataset5_spatial_915.h5ad')
parser.add_argument("--document", type=str, default='dataset5')
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--hidden_size", type=int, default=1024)

parser.add_argument("--noise_std", type=float, default=10)
parser.add_argument("--head", type=int, default=16)
parser.add_argument("--step", type=int, default=1500)
parser.add_argument("--epoch", type=int, default=900)
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()

n_splits = 5
adata_spatial = sc.read_h5ad('datasets/sp/' + args.sp_data)
adata_seq = sc.read_h5ad('datasets/sc/' + args.sc_data)

adata_seq2 = data_augment(adata_seq.copy(), True, noise_std=args.noise_std)
adata_spatial2 = adata_spatial.copy()

sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)

sc.pp.pca(adata_seq2, n_comps=min(10, adata_seq2.n_vars))
sc.pp.neighbors(adata_seq2)
sc.tl.leiden(adata_seq2, resolution=0.4)
adata_seq2 = scale(adata_seq2)
data_seq_array = adata_seq2.X

y = adata_seq2.obs['leiden'].cat.codes.to_numpy(dtype=np.int64)
num_unique = len(np.unique(y))

sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)

sc.pp.pca(adata_spatial2, n_comps=min(10, adata_spatial2.n_vars))
sc.pp.neighbors(adata_spatial2)
sc.tl.leiden(adata_spatial2, resolution=0.4)
adata_spatial2 = scale(adata_spatial2)
data_spatial_array = adata_spatial2.X

sp_y = adata_spatial2.obs['leiden'].cat.codes.to_numpy(dtype=np.int64)


sp_genes = np.array(adata_spatial.var_names)
sp_data = pd.DataFrame(data=data_spatial_array, columns=sp_genes)
sc_data = pd.DataFrame(data=data_seq_array, columns=sp_genes)


def diffusion_impute():
    lr = 1e-6
    depth = 6
    num_epoch = args.epoch
    diffusion_step = args.step
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    head = args.head

    raw_shared_gene = adata_spatial.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(data_spatial_array)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))

        doc = args.document
        save_path = 'IMDt-ckpt/' + doc + '/'
        os.makedirs(save_path, exist_ok=True)
        save_path_prefix = save_path + 'IMDt%d.pt' % (idx)

        cell_num = data_spatial_array.shape[0]
        gene_num = data_spatial_array.shape[1]
        mask = np.ones((gene_num,), dtype='float32')
        gene_ids_test = test_ind
        mask[gene_ids_test] = 0

        seq = data_seq_array
        st_arr = data_spatial_array
        data_seq_masked = seq * mask
        data_spatial_masked = st_arr * mask

        seq = seq * 2 - 1
        data_seq_masked = data_seq_masked * 2 - 1

        data_ary = data_spatial_array
        st_arr = st_arr * 2 - 1
        data_spatial_masked = data_spatial_masked * 2 - 1

        dataloader = get_data_loader(
            seq,
            y,
            batch_size=batch_size,
            is_shuffle=True
        )

        seed = 1202
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = DiT_IMDt(
            input_size=gene_num,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=head,
            classes=num_unique,
            mlp_ratio=4.0,
            dit_type='dit'
        )

        device = torch.device('cuda:2')
        model.to(device)

        model.train()
        if not os.path.isfile(save_path_prefix):
            normal_train_IMDt(
                model,
                dataloader=dataloader,
                lr=lr,
                num_epoch=num_epoch,
                diffusion_step=diffusion_step,
                device=device,
                pred_type='noise',
                mask=mask
            )
            torch.save(model.state_dict(), save_path_prefix)
        else:
            model.load_state_dict(torch.load(save_path_prefix))

        gt = data_spatial_masked
        noise_scheduler = NoiseScheduler(
            num_timesteps=diffusion_step,
            beta_schedule='cosine'
        )

        dataloader = get_data_loader(
            data_spatial_masked,
            sp_y,
            batch_size=batch_size,
            is_shuffle=False
        )

        model.eval()
        imputation = sample_IMDt(
            model,
            device=device,
            dataloader=dataloader,
            noise_scheduler=noise_scheduler,
            mask=mask,
            gt=gt,
            num_step=diffusion_step,
            sample_shape=(cell_num, gene_num),
            is_condi=True,
            model_pred_type='noise',
            is_classifier_guidance=False,
            omega=0.2,
            IOR_jump=1,
            IOR_start=100,
        )

        all_pred_res[:, gene_ids_test] = imputation[:, gene_ids_test]
        idx += 1

    impu = (all_pred_res + 1) / 2
    return impu

Data = args.document
outdir = 'Result/' + Data + '/'
os.makedirs(outdir, exist_ok=True)


diffusion_result = diffusion_impute() # ok
diffusion_result_pd = pd.DataFrame(diffusion_result, columns=sp_genes)
diffusion_result_pd.to_csv(outdir +  '/diffusion_impute.csv',header = 1, index = 1)

def cal_ssim(im1, im2, M):
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1, C2, C3 = (k1 * L) ** 2, (k2 * L) ** 2, (k2 * L) ** 2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return l12 * c12 * s12


def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        result = pd.concat([result, content / content.max()], axis=1)
    return result


def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        result = pd.concat([result, pd.DataFrame(st.zscore(content), columns=[label])], axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        result = pd.concat([result, content / content.sum()], axis=1)
    return result


def logNorm(df):
    return st.zscore(np.log1p(df))


class CalculateMeteics:
    def __init__(self, raw_data, genes_name, impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

        self.impute_count = pd.read_csv(impute_count_file, header=0, index_col=0)
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        self.prefix = prefix
        self.metric = metric

    def SSIM(self, raw, impute, scale='scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw); impute = scale_max(impute)
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                ssim = 0
            else:
                rc = raw.loc[:, label].fillna(1e-20)
                ic = impute.loc[:, label].fillna(1e-20)
                M = max(rc.max(), ic.max())
                ssim = cal_ssim(rc.values.reshape(-1, 1), ic.values.reshape(-1, 1), M)
            result = pd.concat([result, pd.DataFrame(ssim, index=["SSIM"], columns=[label])], axis=1)
        return result

    def SPCC(self, raw, impute, scale=None):
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                spearmanr = 0
            else:
                rc = raw.loc[:, label].fillna(1e-20)
                ic = impute.loc[:, label].fillna(1e-20)
                spearmanr, _ = st.spearmanr(rc, ic)
            result = pd.concat([result, pd.DataFrame(spearmanr, index=["SPCC"], columns=[label])], axis=1)
        return result

    def JS(self, raw, impute, scale='scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw); impute = scale_plus(impute)
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                JS = 1
            else:
                rc = raw.loc[:, label].fillna(1e-20)
                ic = impute.loc[:, label].fillna(1e-20)
                M = (rc + ic) / 2
                JS = 0.5 * st.entropy(rc, M) + 0.5 * st.entropy(ic, M)
            result = pd.concat([result, pd.DataFrame(JS, index=["JS"], columns=[label])], axis=1)
        return result

    def RMSE(self, raw, impute, scale='zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw); impute = scale_z_score(impute)
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                rmse = 1.5
            else:
                rc = raw.loc[:, label].fillna(1e-20)
                ic = impute.loc[:, label].fillna(1e-20)
                rmse = np.sqrt(((rc - ic) ** 2).mean())
            result = pd.concat([result, pd.DataFrame(rmse, index=["RMSE"], columns=[label])], axis=1)
        return result

    def cluster(self, impu, scale=None):
        cpy_x = adata_spatial2.copy()
        cpy_x.X = impu
        sc.tl.pca(adata_spatial2)
        sc.pp.neighbors(adata_spatial2, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(adata_spatial2)
        tmp_adata1 = adata_spatial2
        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x
        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']
        ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
        return pd.DataFrame([[ari, ami, homo, nmi]], columns=["ARI", "AMI", "Homo", "NMI"])

    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        SSIM_gene = self.SSIM(raw, impute)
        Spearman_gene = self.SPCC(raw, impute)
        JS_gene = self.JS(raw, impute)
        RMSE_gene = self.RMSE(raw, impute)
        cluster_result = self.cluster(impute)
        result_gene = pd.concat([Spearman_gene, SSIM_gene, RMSE_gene, JS_gene], axis=0)
        result_gene.T.to_csv(prefix + "_gene_Metrics.txt", sep='\t', header=1, index=1)
        cluster_result.to_csv(prefix + '_cluster_Metrics.txt', sep='\t', header=1, index=1)
        return result_gene


def CalDataMetric(Data):
    print('We are calculating the : ' + Data + '\n')
    metric = ['SPCC', 'SSIM', 'RMSE', 'JS']
    impute_count_dir = outdir
    impute_count = [x for x in os.listdir(impute_count_dir) if x[-3:] == 'csv']
    methods = []
    if not impute_count:
        return
    medians = pd.DataFrame()
    for impute_count_file in impute_count:
        print(impute_count_file)
        if 'result_Tangram.csv' in impute_count_file:
            os.system('mv ' + impute_count_dir + '/result_Tangram.csv ' + impute_count_dir + '/Tangram_impute.csv')
        prefix = impute_count_file.split('_')[0]
        methods.append(prefix)
        prefix = impute_count_dir + '/' + prefix
        impute_count_file = impute_count_dir + '/' + impute_count_file
        CM = CalculateMeteics(data_spatial_array, sp_genes,
                              impute_count_file=impute_count_file,
                              prefix=prefix, metric=metric)
        CM.compute_all()
        median = []
        for j in ['_gene']:
            tmp = pd.read_csv(prefix + j + '_Metrics.txt', sep='\t', index_col=0)
            for m in metric:
                median.append(np.median(tmp[m]))
        median = pd.DataFrame([median], columns=['SPCC(gene)', 'SSIM(gene)', 'RMSE(gene)', 'JS(gene)'])
        clu = pd.read_csv(prefix + '_cluster_Metrics.txt', sep='\t', index_col=0)
        median = pd.concat([median, clu], axis=1)
        medians = pd.concat([medians, median], axis=0)
    metrics = ['SPCC(gene)', 'SSIM(gene)', 'RMSE(gene)', 'JS(gene)', "ARI", "AMI", "Homo", "NMI"]
    medians.columns = metrics
    medians.index = methods
    medians.to_csv(outdir + '/final_result.csv', header=1, index=1)


CalDataMetric(Data)