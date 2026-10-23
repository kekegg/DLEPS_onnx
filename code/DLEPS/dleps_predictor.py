########################################################
# All rights reserved. 
# Author: XIE Zhengwei @ Beijing Gigaceuticals Tech Co., Ltd 
#                      @ Peking University International Cancer Institute
# Contact: xiezhengwei@gmail.com
#
#
########################################################




from os import path

import numpy as np
import pandas as pd
import nltk
import h5py
import onnxruntime as onnxrt

from utils import get_fp, to1hot
import molecule_vae
from molecule_vae import get_zinc_tokenizer
import zinc_grammar
from vectorized_cmap import computecs

# ONNX weights shipped with this repository (relative to code/DLEPS).
DEFAULT_ONNX_DIR = '../../data/onnx'
DEFAULT_ONNX_FILES = ['dleps_0b_0.73.h5.onnx']


class DLEPS(object):
    def __init__(self, setmean = False, reverse = True, base = -2, up_name=None, down_name=None, save_exp = None,
                 onnx_dir = DEFAULT_ONNX_DIR, onnx_files = None):

        
        self.save_exp = save_exp
        self.reverse = reverse
        self.base = base
        self.setmean = setmean
        self.loaded = False
        
        # ONNX inference sessions (Keras / TensorFlow are not required).
        self.sessions = self._load_onnx(onnx_dir, DEFAULT_ONNX_FILES if onnx_files is None else onnx_files)
        self.loaded = True
        self.W = self._get_W()
        A3, self.con = self._get_con()
        self.full_con = A3.dot(self.W)
        
        self.genes=pd.read_table("../../data/gene_info.txt",header=0)
        self.gene_dict=dict(list(zip(self.genes["pr_gene_symbol"], self.genes["pr_gene_id"])))
        
        if up_name:
            self.ups_new = self._get_genes(up_name)
        else:
            self.ups_new = None
            self.reverse = False
            print('DLEPS: No input of up files\n')

        if down_name:
            self.downs_new = self._get_genes(down_name)
        else:
            self.downs_new = None
            self.reverse = False
            print('DLEPS: No input of down files\n')

    def _load_onnx(self, onnx_dir, onnx_files):
        # The exported ONNX graph already contains the whole inference stack
        # (grammar encoder -> VAE sampling -> expression head), so the Keras
        # grammar model ('../../data/vae.hdf5') is not needed any more.
        sessions = []
        for fn in onnx_files:
            p = path.join(onnx_dir, fn)
            if not path.exists(p):
                print('DLEPS: ONNX weights not found at %s\n' % p)
                continue
            sessions.append(onnxrt.InferenceSession(p, providers=['CPUExecutionProvider']))
            print('DLEPS: loaded ONNX weights %s\n' % p)
        if len(sessions) == 0:
            raise FileNotFoundError(
                'DLEPS: no ONNX weights found in "%s" (expected %s). '
                'The ONNX weights are shipped in data/onnx/ of this repository.'
                % (onnx_dir, ', '.join(onnx_files)))
        return sessions

    def _predict_onnx(self, onehot):
        # Run the ONNX graph(s). With several models the predictions are averaged,
        # the same ensemble rule as the AIDock server.
        feats = onehot.astype(np.float32)
        res = []
        for sess in self.sessions:
            inp = sess.get_inputs()[0].name
            out = sess.get_outputs()[0].name
            res.append(sess.run([out], {inp: feats})[0])
        return np.mean(np.array(res), axis=0)
    
    # Map the 978 genes to 12328 genes
    def _get_W(self):
        hf = h5py.File('../../data/denseweight.h5', 'r')
        n1 = hf.get('W')
        W = np.array(n1)
        return W
    
    # The average expression levels for 978 genes
    def _get_con(self):
        benchmark = pd.read_csv('../../data/benchmark.csv')
        A3 = np.concatenate((np.array([1]),benchmark['1.0'].values),axis=0)
        con = benchmark['1.0'].values
        return A3, con
    
    # Get the gene signatures
    def _get_genes(self, fl_name):
        print(fl_name)
        up = pd.read_csv(fl_name,header=None)
        ups= up.values.astype(int)
        print(ups.shape)
        ups = list(np.squeeze(ups))
        ups_new = [i for i in ups if i in list(self.genes["pr_gene_id"])]
        print(ups_new)
        return ups_new
    
    def train(self, smile_train, rna_train, validation_data, epochs=30000, batch_size=512, shuffle=True):
        raise NotImplementedError(
            'This is the ONNX build of DLEPS: the exported graph is inference-only and '
            'cannot be trained further. Train with the original DLEPS (Keras) build and '
            're-export the resulting weights to ONNX.')

    # Only calculate the expression changes, clean smiles required
    def cal_expr(self, smiles, average=True):
        expr = []
        if len(smiles) > 0:
            onehot = to1hot(smiles)
            expr = self._predict_onnx(onehot)
            return expr
        else:
            return []

    # Compute the erichment score
    def comp_cs(self, expr, save_file):
        abs_expr = expr + self.con
        A2 = np.hstack([np.ones([expr.shape[0],1]), abs_expr])
        L12k = A2.dot(self.W)
        if self.setmean:
            L12k_df = pd.DataFrame(L12k, columns=self.genes["pr_gene_id"])
            L12k_df = L12k_df - L12k_df.mean()
        else:
            L12k_delta = L12k-self.full_con
            L12k_df = pd.DataFrame(L12k_delta, columns=self.genes["pr_gene_id"])
        cs = computecs(self.ups_new, self.downs_new, L12k_df.T)
        if save_file:
            L12k_df.T.to_hdf(save_file, key='data')
        return cs[0].values
        
    def predict(self, smiles, average=True, save_onehot=None, load_onehot=None, save_expr = None):
        score = []
        idx = []
        expr = []
        for i in range(len(smiles)):
            score.append(self.base)

        if load_onehot:
            clean_smiles = smiles
        else:
            fps = get_fp(smiles)
            assert len(smiles) == len(fps)
            clean_smiles = []
            clean_fps = []
            nan_smiles = []
            for i in range(len(fps)):
                if np.isnan(sum(fps[i])):
                    nan_smiles.append(smiles[i])
                else:
                    clean_smiles.append(smiles[i])
                    clean_fps.append(fps[i])
                    idx.append(i)
            clean_fps = np.array(clean_fps)

        if len(clean_smiles) > 0:
            if load_onehot:
                fss = load_onehot.split('.')
                if fss[-1] == 'npz':
                    onehotz = np.load(load_onehot)
                    print(onehotz.files)
                    onehot = onehotz[onehotz.files[0]]
                else:
                    onehot = np.load(load_onehot)
                head = '.'.join(fss[:-1])
                idx = np.load(head+'_idx.npy').astype(int)
            else:
                onehot = to1hot(clean_smiles)
                if save_onehot:
                    np.save(save_onehot,onehot)
                    np.save(save_onehot+'_idx',idx)
            if self.loaded:
                expr = self._predict_onnx(onehot)
            print('DLEPS: 978 signatures obtained\n\n')
            step_size = 50000
            
            #If input SMILES is too large, split the files 
            if expr.shape[0] > step_size:
                cs_arr = np.zeros(expr.shape[0])
                num = int(expr.shape[0]/(step_size*1.0))
                for i in range(num):
                    print(i,num,sep=':')
                    cur_expr = expr[i*step_size:(i+1)*step_size]
                    if save_expr:
                        cur_cs = self.comp_cs(cur_expr, save_expr+'SMILES_L12k_'+str(i)+'.h5')
                    else:
                        cur_cs = self.comp_cs(cur_expr, None)
                    for j in range(cur_cs.shape[0]):
                        cs_arr[i*step_size+j] = cur_cs[j]

                cur_expr = expr[num*step_size:expr.shape[0]]
                if save_expr:
                    cur_cs = self.comp_cs(cur_expr, save_expr+'SMILES_L12k_'+str(num)+'.h5')
                else:
                    cur_cs = self.comp_cs(cur_expr,None)

                for j in range(cur_cs.shape[0]):
                    cs_arr[num*step_size+j] = cur_cs[j]
            else:
                abs_expr = expr + self.con
                A2 = np.hstack([np.ones([expr.shape[0],1]), abs_expr])
                L12k = A2.dot(self.W)
                if self.setmean:
                    L12k_df = pd.DataFrame(L12k, columns=self.genes["pr_gene_id"])
                    L12k_df = L12k_df - L12k_df.mean()
                else:
                    L12k_delta = L12k-self.full_con
                    L12k_df = pd.DataFrame(L12k_delta, columns=self.genes["pr_gene_id"])
                if self.save_exp:
                    L12k_df.to_csv(self.save_exp)

                print('DLEPS: 12328 gene changes obtained\n\n')
                if save_expr:
                    cs_arr = self.comp_cs(expr,save_expr)
                else:
                    cs = computecs(self.ups_new, self.downs_new, L12k_df.T)
                    cs_arr = cs[0].values
                print('DLEPS: Enrichment scores were calculated\n\n')

        for i in range(len(score)):
            if self.reverse:
                score[idx[i]] = cs_arr[i]*(-1)
            else:
                score[idx[i]] = cs_arr[i]

        return score
