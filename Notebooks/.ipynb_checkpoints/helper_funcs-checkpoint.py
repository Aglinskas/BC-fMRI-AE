import os
import numpy as np
from matplotlib import pyplot as plt
import umap
from IPython import display
import time
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.decomposition import PCA
from umap import UMAP


default_keys = ['ADOS_Total','ADOS_Social','DSMIVTR','AgeAtScan','Sex','ScannerID','ScanSiteID','FIQ']


class vae_data_loader():
    ''' this is the info'''
    def __init__(self,data_dir,df,batch_size=32):
        #self.data = data
        #self.n = data.shape[0]
        self.n = len(df)
        self.data_dir = data_dir
        self.epoch = -1
        self.batch_size = batch_size
        self.df = df
        
        self.new_epoch()
        self.n_batches = int(self.n/self.batch_size)
        
    def new_epoch(self):
        #self.df = self.df.iloc[np.random.permutation(np.arange(self.n))]
        #self.df.index = np.arange(self.n)
        self.idxs = np.arange(self.n)
        
        #self.asd_idxs = np.random.permutation(self.asd_idxs)
        self.idxs = np.random.permutation(self.idxs)
        
        self.epoch += 1
        self.b = 0
        
        
    def get_batch(self):
        self.b += 1
        
        if self.b==self.n_batches:
            self.new_epoch()
        
       # self.batch_asd_idx = self.asd_idxs[np.arange(self.b*self.batch_size,self.b*self.batch_size+self.batch_size)]
        self.batch_idx = self.idxs[np.arange(self.b*self.batch_size,self.b*self.batch_size+self.batch_size)]
        #self.batch_asd = np.array([np.load(os.path.join(self.data_dir,val+'.npy')) for val in self.df['bids_folder'].values[self.batch_asd_idx]])
        self.batch = np.array([np.load(os.path.join(self.data_dir,val+'.npy')) for val in self.df['bids_folder'].values[self.batch_idx]])
        self.batch_df = self.df.iloc[self.batch_idx,:]
        
        self.batch = self.batch/2
        #self.batch_td = self.batch_td/2
        
    
        return self.batch,self.batch_df
    
    

class cvae_data_loader():
    ''' this is the info'''
    def __init__(self,data_dir,df,batch_size=32):
        #self.data = data
        #self.n = data.shape[0]
        self.n = len(df)
        self.data_dir = data_dir
        self.epoch = -1
        self.batch_size = batch_size
        self.df = df
        
        self.new_epoch()
        self.n_batches = int(np.floor(min((len(self.asd_idxs),len(self.td_idxs)))/self.batch_size))
        
    def new_epoch(self):
        #self.df = self.df.iloc[np.random.permutation(np.arange(self.n))]
        #self.df.index = np.arange(self.n)
        self.asd_idxs = np.nonzero((self.df['DX_GROUP'].values==1))[0]
        self.td_idxs = np.nonzero((self.df['DX_GROUP'].values==2))[0]
        
        self.asd_idxs = np.random.permutation(self.asd_idxs)
        self.td_idxs = np.random.permutation(self.td_idxs)
        
        self.epoch += 1
        self.b = 0
        
        
    def get_batch(self):
        self.b += 1
        
        if self.b==self.n_batches:
            self.new_epoch()
        
        
        self.batch_asd_idx = self.asd_idxs[np.arange(self.b*self.batch_size,self.b*self.batch_size+self.batch_size)]
        self.batch_td_idx = self.td_idxs[np.arange(self.b*self.batch_size,self.b*self.batch_size+self.batch_size)]
        self.batch_asd = np.array([np.load(os.path.join(self.data_dir,val+'.npy')) for val in self.df['bids_folder'].values[self.batch_asd_idx]])
        self.batch_td = np.array([np.load(os.path.join(self.data_dir,val+'.npy')) for val in self.df['bids_folder'].values[self.batch_td_idx]])
        self.batch_df = self.df.iloc[np.hstack((self.batch_asd_idx,self.batch_td_idx)),:]
        
        self.batch_asd = self.batch_asd/2
        self.batch_td = self.batch_td/2
        
    
        return self.batch_asd,self.batch_td,self.batch_df

def fit_rsa(rdm_data,rdm_model):
    return np.corrcoef(get_triu(rdm_data),get_triu(rdm_model))[0,1]
        
def make_RDM(inVec,data_scale='ratio',metric='euclidean'):
    vec = inVec
    vec = (vec - min(vec.flatten())) / (max(vec.flatten())-min(vec.flatten()))
    
    if np.ndim(inVec)==1: # must be at least 2D
        vec = np.vstack((vec,np.zeros(vec.shape))).transpose()
                   
    mat = squareform(pdist(vec,metric=metric).transpose())
    
    #vec = (vec - min(vec)) / (max(vec)-min(vec))
    #mat = (mat - min(mat)) / (max(mat)-min(mat))

    if data_scale=='ordinal':
        mat[mat!=0]=1 # Make into zeros and ones
        
    return mat


def get_triu(inMat):
    #inMat = rdm_bg

    assert np.ndim(inMat)==2, 'not 2 dim, wtf'
    assert inMat.shape[0]==inMat.shape[1], 'not a square'

    n = inMat.shape[0]
    triu_vec = inMat[np.triu_indices(n=n,k=1)]

    #assert (squareform(triu_vec)==inMat).sum()/(n**2)>.9, 'unfaithful triu'
    return triu_vec


def cvae_dashboard(epoch,b_idx,i,cvae,z_encoder,s_encoder,loss,all_rsas,c_sim,df,patient_batch,control_batch,batch_rsas,red='UMAP'):
    
    if red=='PCA':
        reducer = PCA
    elif red=='UMAP':
        reducer = UMAP
    else:
        reducer = UMAP
        
    #prediction = cvae.predict(patient_batch)    
    predictions = cvae.predict([patient_batch,control_batch])
    #predictions = np.concatenate((predictions[0],predictions[1]),axis=0)
    prediction = predictions[0]
    
    if np.mod(i,1)==0:        
        pc_s = reducer(n_components=2).fit_transform(s_encoder.predict(np.concatenate((patient_batch,control_batch),axis=0))[2])
        pc_z = reducer(n_components=2).fit_transform(z_encoder.predict(np.concatenate((patient_batch,control_batch),axis=0))[2])
        
        global pc_s_asd, pc_s_td, pc_z_asd, pc_z_td

        pc_s_asd = pc_s[0:patient_batch.shape[0],:]
        pc_s_td = pc_s[patient_batch.shape[0]::,:]

        pc_z_asd = pc_z[0:patient_batch.shape[0],:]
        pc_z_td = pc_z[patient_batch.shape[0]::,:]
        
        #all_rsas.append(get_batch_rsas(z_encoder,s_encoder,df,n=50))
        
    cmat_actual = np.corrcoef(np.vstack((patient_batch.reshape(patient_batch.shape[0],-1),control_batch.reshape(control_batch.shape[0],-1))))
    cmat_pred = np.corrcoef(np.vstack((predictions[0].reshape(predictions[0].shape[0],-1),predictions[1].reshape(predictions[1].shape[0],-1))))
    c_sim.append(np.corrcoef(get_triu(cmat_pred),get_triu(cmat_actual))[0,1])

    
    ### PROGRESS PLOTTING
    display.clear_output(wait=True);
    display.display(plt.gcf());
    #Organise figure
    ncols = 4;nrows=5
    if np.mod(i,5)==0:
        plt.close()
    plt.subplots(nrows,ncols,figsize=(15,15)); # MAKE THE FIGURE
    
    
    ##### SUBPLOT 1 & 2 ##### 

    plt.subplot(nrows,ncols/2,1) # PLOT LOSS
    xs = np.arange(len(loss))+1
    m,b = np.polyfit(xs,loss,deg=1)
    plt.plot(loss)
    plt.plot(xs, m*xs + b)
    plt.title(f'Epoch {epoch} batch {b_idx} | Loss {loss[-1]:.2f}, beta: {m:.4f}')
    
    ##### SUBPLOT 3 ##### 
    plt.subplot(nrows,ncols,3) # PLOT LOSS LAST 50
    hb = 50
    if len(loss)>hb:
        plot_loss = loss[-hb::]
        xs = np.arange(len(plot_loss))
        m,b = np.polyfit(xs,plot_loss,deg=1)
        plt.plot(plot_loss)
        plt.plot(xs, m*xs + b)
        #plt.title(hist)
        plt.title(f'Loss last {hb} it, beta {m:.4f}')
        
    ##### SUBPLOT 4 ##### 
    plt.subplot(nrows,ncols,4)
    plt.hist(prediction[0,:,:,:,0].flatten(),alpha=.5)
    plt.hist(patient_batch[0,:,:,:,0].flatten(),alpha=.5)
    plt.legend(['predicted','actual'])
    plt.title('in/out histograms')
    
    ##### SUBPLOT 5 ##### 
    plt.subplot(nrows,ncols,5) #RSA over time
    plt.plot(c_sim)
    plt.title(f'in/out RSA: {c_sim[-1].round(2)}')
    
    if len(c_sim)>5: # PLOT LS LINE
        xs = np.arange(len(c_sim))+1
        m,b = np.polyfit(xs,c_sim,deg=1)
        plt.plot(xs, m*xs + b)
        plt.title(f'in/out RSA: {c_sim[-1].round(2)}, b={m:.4f}')
        

    ##### SUBPLOT 6 ##### 
    plt.subplot(nrows,ncols,6)
    if len(c_sim)>hb:
        #plot_loss = loss[-hb::]
        xs = np.arange(len(c_sim[-hb::]))
        m,b = np.polyfit(xs,c_sim[-hb::],deg=1)
        plt.plot(c_sim[-hb::])
        plt.plot(xs, m*xs + b)
        #plt.title(hist)
        plt.title(f'in/outRSA last {hb} it, b={m:.4f}')

    ##### SUBPLOT 7 ##### 
    plt.subplot(nrows,ncols,7)
    lbls = ['age','sex','fiq','dsm','site','ados'];
    xs = np.arange(len(lbls));
    plt.bar(xs,batch_rsas[0:6]);
    plt.xticks(xs,labels=lbls);
    plt.title('S RSA')
    
    ##### SUBPLOT 8 ##### 
    plt.subplot(nrows,ncols,8)
    lbls = ['age','sex','fiq','dsm','site','ados'];
    xs = np.arange(len(lbls));
    plt.bar(xs,batch_rsas[6::]);
    plt.xticks(xs,labels=lbls);
    plt.title('Z RSA')

        
    ##### SUBPLOT 9 ##### 
    plt.subplot(nrows,ncols,9)
    plt.scatter(pc_s_asd[:,0],pc_s_asd[:,1],s=100,alpha=.5);plt.title(f'batch latent S rep. {red}')
    plt.scatter(pc_s_td[:,0],pc_s_td[:,1],s=100,alpha=.5);plt.title(f'batch latent S rep. {red}')
    plt.legend(['ASD','TD'])
    
    ##### SUBPLOT 10 ##### 
    plt.subplot(nrows,ncols,10)
    plt.scatter(pc_z_asd[:,0],pc_z_asd[:,1],s=100,alpha=.5);plt.title(f'batch latent Z rep. {red}')
    plt.scatter(pc_z_td[:,0],pc_z_td[:,1],s=100,alpha=.5);plt.title(f'batch latent Z rep. {red}')
    plt.legend(['ASD','TD'])
    
    ##### SUBPLOT 11 ##### 
    plt.subplot(nrows,ncols,11)
    sns.heatmap(cmat_actual,xticklabels=[],yticklabels=[])
    plt.title('input RSA')
        
    ##### SUBPLOT 12 ##### 
    plt.subplot(nrows,ncols,12)
    sns.heatmap(cmat_pred,xticklabels=[],yticklabels=[])
    plt.title('output RSA')
  
    ##### SUBPLOT 13 #####     
    plt.subplot(nrows,ncols,13)
    plt.imshow(np.rot90(patient_batch[0,:,16,:,0]))
    plt.xticks([]);plt.yticks([]);plt.title('actual')

    ##### SUBPLOT 14 #####     
    plt.subplot(nrows,ncols,14)
    plt.imshow(np.rot90(prediction[0,:,16,:,0]))
    plt.xticks([]);plt.yticks([]);plt.title('predicted')
    ##### SUBPLOT 15 #####     
    plt.subplot(nrows,ncols,15)
    plt.imshow(np.rot90(patient_batch[0,16,:,:,0]))
    plt.xticks([]);plt.yticks([]);plt.title('actual')
    ##### SUBPLOT 16 #####                                             
    plt.subplot(nrows,ncols,16)
    plt.imshow(np.rot90(prediction[0,16,:,:,0]))
    plt.xticks([]);plt.yticks([]);plt.title('predicted')
    ##### SUBPLOT 17 #####     
    plt.subplot(nrows,ncols,17)
    plt.imshow(np.rot90(patient_batch[0,6,:,:,0]))
    plt.xticks([]);plt.yticks([]);plt.title('actual')
    ##### SUBPLOT 18 #####                                             
    plt.subplot(nrows,ncols,18)
    plt.imshow(np.rot90(prediction[0,6,:,:,0]))
    plt.xticks([]);plt.yticks([]);plt.title('predicted')
    ##### SUBPLOT 19 #####     
    plt.subplot(nrows,ncols,19)
    plt.imshow(patient_batch[0,:,:,16,0])
    plt.xticks([]);plt.yticks([]);plt.title('actual')
    ##### SUBPLOT 20 #####     
    plt.subplot(nrows,ncols,20)
    plt.imshow(prediction[0,:,:,16,0])
    plt.xticks([]);plt.yticks([]);plt.title('predicted')
    
    plt.tight_layout()
    plt.show()
    #sys.stdout.flush()
    
    return c_sim,all_rsas




def get_batch_rsas(data_dir,df,z_encoder,s_encoder,batch_size=400):
    
    temp_loader = cvae_data_loader(data_dir=data_dir, df=df, batch_size=batch_size)
    temp_batch_asd,temp_batch_td,temp_batch_df = temp_loader.get_batch()


    asd_idx =  ~np.isnan(temp_batch_df['ADOS_TOTAL'].values) * (temp_batch_df['DX_GROUP'].values==1) * ~np.isnan(temp_batch_df['FIQ'].values)

    z_embedding = z_encoder.predict(temp_batch_asd[asd_idx[0:batch_size],:,:,:,:])[2]
    s_embedding = s_encoder.predict(temp_batch_asd[asd_idx[0:batch_size],:,:,:,:])[2]

    rdm_age = make_RDM(temp_batch_df['AGE_AT_SCAN'].values[asd_idx])
    rdm_sex = make_RDM(temp_batch_df['SEX'].values[asd_idx],data_scale='ordinal')
    rdm_fiq = make_RDM(temp_batch_df['FIQ'].values[asd_idx])
    rdm_dsm = make_RDM(temp_batch_df['DSM_IV_TR'].values[asd_idx],data_scale='ordinal')
    rdm_site = make_RDM(temp_batch_df['site'].values[asd_idx],data_scale='ordinal')
    rdm_adosTotal = make_RDM(temp_batch_df['ADOS_TOTAL'].values[asd_idx])

    rdm_z = make_RDM(z_embedding)
    rdm_s = make_RDM(s_embedding)

    batch_rsas = list()

    batch_rsas.append(fit_rsa(rdm_s,rdm_age))
    batch_rsas.append(fit_rsa(rdm_s,rdm_sex))
    batch_rsas.append(fit_rsa(rdm_s,rdm_fiq))
    batch_rsas.append(fit_rsa(rdm_s,rdm_dsm))
    batch_rsas.append(fit_rsa(rdm_s,rdm_site))
    batch_rsas.append(fit_rsa(rdm_s,rdm_adosTotal))

    batch_rsas.append(fit_rsa(rdm_z,rdm_age))
    batch_rsas.append(fit_rsa(rdm_z,rdm_sex))
    batch_rsas.append(fit_rsa(rdm_z,rdm_fiq))
    batch_rsas.append(fit_rsa(rdm_z,rdm_dsm))
    batch_rsas.append(fit_rsa(rdm_z,rdm_site))
    batch_rsas.append(fit_rsa(rdm_z,rdm_adosTotal))
    
    return batch_rsas


























































































# def get_weights(fdir=None):
#     if not fdir:
#         fdir = '/mmfs1/data/aglinska/tf_outputs/CVAE/'
#     items = os.listdir(fdir)
#     items = [item.split('.')[0] for item in items]
#     items = [item.replace('_loss','') for item in items]
#     items = np.unique(items)
#     items=items[items!='']
#     items.sort()

#     for i in range(len(items)):
#         print(f"{i:02d} | {items[i]}")
        
#     return items


# def cscatter(spaces,v=None,c=None,clim=None,clbl=None,legend=None,return_axes=False):
#     space_lbls = ['Background','Salient','VAE']

#     if type(v)==type(None):
#         v = np.repeat(True,len(spaces[0]))
        
#     fig = plt.figure(figsize=(12,4))
#     for i in range(len(spaces)):
#         plt.subplot(1,3,i+1)
        
#         if type(c)!=type(None) and len(np.unique(c)) > 10: # continus colourbar
#             #print('continuues colourbar')
            
#             plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
#             if type(clim)==type(None): #if clim not passed, 
#                 clim = (min(c),max(c)) # calc min max
#             plt.clim(clim[0],clim[1]) # do clim regardless
                
#             cbar = plt.colorbar()
#             cbar.ax.set_ylabel(clbl,rotation=270,labelpad=20,fontsize=16,fontweight='bold')    
                
#         elif type(c)!=type(None) and len(np.unique(c)) < 10: # categorical colourbar
#             #print('categorical colourbar')
#             for j in np.unique(c):
#                 plt.scatter(spaces[i][c[v]==j,0],spaces[i][c[v]==j,1],alpha=.5)
                    
#             if type(legend)==type(None):
#                 legend = [str(i) for i in np.unique(c)]    
#             plt.legend(legend)

#         else:
#            #print('else')
#             plt.scatter(spaces[i][v,0],spaces[i][v,1])
            
        
#         #plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
#         plt.xlabel('latent dim. 1');plt.ylabel('latent dim. 2')
#         plt.title(space_lbls[i])

#     plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=.3,hspace=None,) 
#     print(sum(v))
    
#     if return_axes:
#         return fig
    
    
# def get_batch_idx(df,batch_size = 64):

#     sub_scan_site = df['ScanSite'].values
#     scanning_sites = np.unique(sub_scan_site)

#     nsites = len(scanning_sites)

#     this_site = np.random.randint(low=0,high=nsites)


#     site_asd = (sub_scan_site==scanning_sites[this_site]) * (df['DxGroup'].values==1)
#     site_td = (sub_scan_site==scanning_sites[this_site]) * (df['DxGroup'].values==2)

#     asd_idx = np.nonzero(site_asd)[0]
#     td_idx = np.nonzero(site_td)[0]

#     while len(asd_idx) < batch_size: #if not enough copy over
#         asd_idx = np.hstack((asd_idx,asd_idx))

#     while len(td_idx) < batch_size: #if not enough copy over
#         td_idx = np.hstack((td_idx,td_idx))

#     assert len(np.unique(df.iloc[asd_idx]['Subject Type'].values)),'subject batch selection messed up'
#     assert len(np.unique(df.iloc[asd_idx]['ScanSite'].values)),'subject batch selection messed up'
#     assert len(np.unique(df.iloc[asd_idx]['ScannerType'].values)),'subject batch selection messed up'

#     assert len(np.unique(df.iloc[td_idx]['Subject Type'].values)),'subject batch selection messed up'
#     assert len(np.unique(df.iloc[td_idx]['ScanSite'].values)),'subject batch selection messed up'
#     assert len(np.unique(df.iloc[td_idx]['ScannerType'].values)),'subject batch selection messed up'
    
#     assert ~any([a in td_idx for a in asd_idx]),'you f***ed up'
#     assert ~any([t in asd_idx for t in td_idx]),'you f***ed up'
    
#     np.random.shuffle(asd_idx)
#     np.random.shuffle(td_idx)

#     asd_idx = asd_idx[0:batch_size]
#     td_idx = td_idx[0:batch_size]

#     return asd_idx,td_idx


# def dim_reduce(z,method='UMAP'):
    
#     if method=='UMAP':
#         reducer = umap.UMAP()
#         #reducer = ParametricUMAP()
#     else:
#         reducer = PCA(n_components=2)
        
#     tiny = reducer.fit_transform(z)
    
#     return tiny

# def cscatter(spaces,v=None,c=None,clim=None,clbl=None,legend=None):
#     space_lbls = ['Background','Salient','VAE']

#     if type(v)==type(None):
#         v = np.repeat(True,len(spaces[0]))
        
#     plt.figure(figsize=(12,4))
#     for i in range(len(spaces)):
#         plt.subplot(1,3,i+1)
        
#         if type(c)!=type(None) and len(np.unique(c)) > 10: # continus colourbar
#             #print('continuues colourbar')
            
#             plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
#             if type(clim)==type(None): #if clim not passed, 
#                 clim = (min(c),max(c)) # calc min max
#             plt.clim(clim[0],clim[1]) # do clim regardless
                
#             cbar = plt.colorbar()
#             cbar.ax.set_ylabel(clbl,rotation=270,labelpad=20,fontsize=16,fontweight='bold')    
                
#         elif type(c)!=type(None) and len(np.unique(c)) < 10: # categorical colourbar
#             #print('categorical colourbar')
#             for j in np.unique(c):
#                 plt.scatter(spaces[i][c[v]==j,0],spaces[i][c[v]==j,1],alpha=.5)
                    
#             if type(legend)==type(None):
#                 legend = [str(i) for i in np.unique(c)]    
#             plt.legend(legend)

#         else:
#            #print('else')
#             plt.scatter(spaces[i][v,0],spaces[i][v,1])
            
        
#         #plt.scatter(spaces[i][v,0],spaces[i][v,1],c=c)
#         plt.xlabel('latent dim. 1');plt.ylabel('latent dim. 2')
#         plt.title(space_lbls[i])

#     plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=.3,hspace=None,) 
#     #print(sum(v))
#     plt.show()
    
    
    
    
    
    
# def get_spaces(ABIDE_data,z_encoder,s_encoder,w=2,method='UMAP'):
    
#     encs = [z_encoder.predict, s_encoder.predict]
#     bg_space = np.array(encs[0](ABIDE_data)[w])
#     sl_space = np.array(encs[1](ABIDE_data)[w])

#     if bg_space.shape[1]>2:
#         bg_space = dim_reduce(bg_space,method=method)
#         sl_space = dim_reduce(sl_space,method=method)
#     return bg_space,sl_space


# def plot_sweep(ABIDE_data,z_encoder,s_encoder,cvae_decoder,wspace='z',l=5,w=2):

#     z = z_encoder.predict(ABIDE_data)
#     s = s_encoder.predict(ABIDE_data)

#     z_lin = np.linspace(z[2].min(axis=0),z[2].max(axis=0),l)
#     s_lin = np.linspace(s[2].min(axis=0),s[2].max(axis=0),l)
    
#     nrows = l;ncols = l;c = 0
    
#     for i in range(l):
#         for j in range(l):
#             c+=1
#             plt.subplot(nrows,ncols,c)
#             vec_z = z_lin[i,:]
#             vec_s = s_lin[i,:]
#             vec_0 = np.zeros(vec_s.shape)

#             if wspace=='z':
#                 vec3 = np.hstack((vec_z,vec_0))
#             elif wspace=='s':
#                 vec3 = np.hstack((vec_0,vec_s))
#             else:
#                 #vec3 = np.hstack((vec_z,vec_s))
#                 vec3 = np.hstack((vec_z,vec_s))

#             plt.imshow(cvae_decoder.predict(np.vstack((vec3,vec3)))[0,:,:,32,0])
#             plt.xticks([]);plt.yticks([]);
#     plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    
# def plot_four(DX_batch,TD_batch,z_encoder,s_encoder,cvae_decoder,cvae,idx=0,v=2,s=11,k=40,axis='ax'):
    
#     #im_in = [DX_batch,TD_batch][np.random.choice([0,1])]
#     im_in = [DX_batch,TD_batch][idx]
#     _zeros = np.zeros(s_encoder(im_in)[2].shape)

#     #v = 2
#     cvae_sal_vec = np.hstack((_zeros,s_encoder(im_in)[v]))
#     cvae_bg_vec = np.hstack((z_encoder(im_in)[v],_zeros))
    
#     if idx==1:
#         cvae_full_vec = np.hstack((z_encoder(im_in)[v],s_encoder(im_in)[v]))
#     elif idx==0:
#         cvae_full_vec = cvae_bg_vec
    
#     #cvae_sal_vec = np.hstack((s_encoder(im_in)[v],_zeros))
#     #cvae_bg_vec = np.hstack((_zeros,z_encoder(im_in)[v]))
    
#     if axis=='ax':
#         plot_im_input = im_in[s,:,:,k]
#         plot_im_sal = cvae_decoder(cvae_sal_vec)[s,:,:,k,0]
#         plot_im_bg = cvae_decoder(cvae_bg_vec)[s,:,:,k,0]
#         #plot_im_recon = cvae.predict([DX_batch,TD_batch])[idx][s,:,:,k,0]
#         plot_im_recon = cvae_decoder(cvae_full_vec)[s,:,:,k,0]
#     elif axis=='sag':
#         plot_im_input = im_in[s,k,:,:]
#         plot_im_sal = cvae_decoder(cvae_sal_vec)[s,k,:,:,0]
#         plot_im_bg = cvae_decoder(cvae_bg_vec)[s,k,:,:,0]
#         #plot_im_recon = cvae.predict([DX_batch,TD_batch])[idx][s,k,:,:,0]
#         plot_im_recon = cvae_decoder(cvae_full_vec)[s,k,:,:,0]
#     elif axis=='cor':
#         plot_im_input = im_in[s,:,k,:]
#         plot_im_sal = cvae_decoder(cvae_sal_vec)[s,:,k,:,0]
#         plot_im_bg = cvae_decoder(cvae_bg_vec)[s,:,k,:,0]
#         #plot_im_recon = cvae.predict([DX_batch,TD_batch])[idx][s,:,k,:,0]
#         plot_im_recon = cvae_decoder(cvae_full_vec)[s,:,k,:,0]
        

        
#     plt.figure(figsize=np.array((4*4,4))*.5)
#     plt.subplot(1,4,1)
#     plt.imshow(plot_im_input);plt.xticks([]);plt.yticks([]);plt.title('input')
#     plt.subplot(1,4,2)
#     plt.imshow(plot_im_recon);plt.xticks([]);plt.yticks([]);plt.title('reconstruction')
#     plt.subplot(1,4,3)
#     plt.imshow(plot_im_sal);plt.xticks([]);plt.yticks([]);plt.title('salient')
#     plt.subplot(1,4,4)
#     plt.imshow(plot_im_bg);plt.xticks([]);plt.yticks([]);plt.title('background')
    
#     plt.show()
    
    
# def str_to_ordinal(inVec):
#     #inVec = df['ScannerType'].values
#     u = np.unique(inVec)
#     evec = np.zeros(inVec.shape)
#     for i in range(len(u)):
#         evec[inVec==u[i]]=i
        
#     return evec


# def plot_cvae_silhouettes(ABIDE_data,z_encoder,s_encoder,df,sub_slice,keys=None,l=5,w=2):
#     from sklearn.metrics import silhouette_score
    
#     z = z_encoder.predict(ABIDE_data)[w]
#     s = s_encoder.predict(ABIDE_data)[w]
    
    
#     if not keys:
#         keys = ['ADOS_Total','ADOS_Social','DSMIVTR','AgeAtScan','Sex','ScannerID','ScanSiteID','FIQ']
    
#     #l = 5 # How many bings
#     arr = np.zeros((len(keys),2))
#     mat = np.zeros((len(keys),l))
#     for i in range(len(keys)):
#         vec = df[keys[i]].values
#         v = (vec!=-9.999e+03) * ~pd.isnull(vec) * sub_slice
#         vecv = vec[v]
#         vecv = np.digitize(vec[v],np.linspace(vec[v].min(),vec[v].max(),l))
#         arr[i,0]= silhouette_score(z[v],vecv)
#         arr[i,1]= silhouette_score(s[v],vecv)

#     plot_df = pd.DataFrame(arr,index=keys,columns=['BG','SL'])
#     ax = plot_df.plot.bar(rot=45)
#     ax.legend()

#     plt.show()
    
    
# def get_sil_diff(ABIDE_data,z_encoder,s_encoder,df,sub_slice,keys=None,l=5,w=2):
    
#     z = z_encoder.predict(ABIDE_data)[w]
#     s = s_encoder.predict(ABIDE_data)[w]

#     if not keys:
#         keys = ['ADOS_Total','ADOS_Social','DSMIVTR','AgeAtScan','Sex','ScannerID','ScanSiteID','FIQ']
#         #keys = ['AgeAtScan','ScannerID','ScanSiteID','FIQ']

#     arr = np.zeros((len(keys),2))
#     #mat = np.zeros((len(keys),l))
#     for i in range(len(keys)):
#         vec = df[keys[i]].values
#         v = (vec!=-9.999e+03) * ~pd.isnull(vec) * sub_slice
#         vecv = vec[v]
#         vecv = np.digitize(vec[v],np.linspace(vec[v].min(),vec[v].max(),l))
#         arr[i,0]= silhouette_score(z[v],vecv)
#         arr[i,1]= silhouette_score(s[v],vecv)

#     dif = arr[:,1]-arr[:,0]
    
#     return dif


# def plot_cvae_dif_mat(ABIDE_data,z_encoder,s_encoder,df,sub_slice,keys=None,l=5,w=2,ax=None,title=None,return_arr=False):
    
#     if not keys:
#         keys = default_keys
        
#     dmat = np.array([get_sil_diff(ABIDE_data,z_encoder,s_encoder,df,sub_slice,keys=keys,l=l,w=w) for i in range(10)])
    
#     ys = dmat.mean(axis=0)
#     yerr = dmat.std(axis=0)
#     xs = np.arange(dmat.shape[1])
    
#     if not ax:
#         plt.bar(xs,ys);
#         plt.errorbar(xs,ys,yerr,fmt='r.');
#         plt.xticks(xs,labels=keys,rotation=90);
#         plt.ylabel('SL-BG Silhouette Difference')
#     else:
#         ax.bar(xs,ys);
#         ax.errorbar(xs,ys,yerr,fmt='r.');
#         ax.set_xticks(xs);
#         ax.set_xticklabels(keys,rotation=90)
#         ax.set_ylabel('SL-BG Silhouette Difference')
#         ax.set_title(title)
    
#     if return_arr:
#         return dmat
    
        
#     plt.show()
# # def plot_cvae_dif_mat(ABIDE_data,z_encoder,s_encoder,df,sub_slice,keys=None,l=5,w=2):
    
# #     if not keys:
# #         keys = default_keys
        
# #     dmat = np.array([get_sil_diff(ABIDE_data,z_encoder,s_encoder,df,sub_slice,keys=keys,l=l,w=w) for i in range(10)])

# #     ys = dmat.mean(axis=0)
# #     yerr = dmat.std(axis=0)
# #     xs = np.arange(dmat.shape[1])

# #     plt.bar(xs,ys);
# #     plt.errorbar(xs,ys,yerr,fmt='r.');
# #     plt.xticks(xs,labels=keys,rotation=90);
# #     plt.ylabel('SL-BG Silhouette Difference')
# #     plt.show()
    
    
# # Progress Plotting Functions
# def cvae_query(ABIDE_data,s_encoder,z_encoder,cvae_decoder):
#     i = 0
#     n = 50
#     v_sl = s_encoder.predict(ABIDE_data[0:n,:,:,:])[i]#[0,:]
#     v_bg = z_encoder.predict(ABIDE_data[0:n,:,:,:])[i]#[0,:]
#     #v = np.hstack((v_sl,v_bg))
#     v = np.hstack((v_bg,v_sl))
#     latent_vec = v;
#     out = cvae_decoder.predict(latent_vec)

#     im = out[:,:,:,:,0]
#     im1 = ABIDE_data[0:n,:,:,:]
#     ss = ((im-im1)**2).sum()

#     return im[0,:,:,40],im1[0,:,:,40],ss

# def net_plot(im,im1):
#     plt.subplot(1,2,1);
#     plt.imshow(im1);
#     plt.subplot(1,2,2);
#     plt.imshow(im);

# def plot_trainProgress(loss,im,im1):

#     display.clear_output(wait=True);
#     display.display(plt.gcf());
#     #time.sleep(1.0)

#     plt.figure(figsize=np.array((7,5)) );

#     plt.subplot(2,2,1);
#     plt.imshow(im1);plt.xticks([]);plt.yticks([]);
#     plt.title('image')

#     plt.subplot(2,2,3);
#     plt.imshow(im);plt.xticks([]);plt.yticks([]);
#     plt.title('reconstruction')

#     # Last 1000
#     plt.subplot(2,2,2);
    
#     def moving_average(x, w):
#         return np.convolve(x, np.ones(w), 'valid') / w

#     ss = int(len(loss)*1/3)
#     plt.plot(loss[ss::],alpha=.3)
#     plt.plot(moving_average(loss[ss::], 100))

#     # Last 100
#     plt.subplot(2,2,4);
#     n = 1000
#     if len(loss)>n:
#         plt.plot(loss[-n::]);plt.title(f'loss: last {n} iteration');
#         plt.plot(moving_average(loss[-n::], 10))
#     else:
#         plt.plot(loss);plt.title('overall loss');

#     plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.15, hspace=.45);

#     plt.show();
    
# def fit_rsa(inVec,ABIDE_data,g,encs,data_scale='ratio',metric='correlation'):
#     # encs = [z_encoder,s_encoder,encoder]

#     v = g*~np.isnan(inVec)
#     vec_model = get_triu(make_RDM(inVec[v],data_scale=data_scale))
    
#     vecs = [get_triu(make_RDM(enc.predict(ABIDE_data[v,:,:,:])[2])) for enc in encs]
#     #f = [np.corrcoef(vec_model,vec)[0,1] for vec in vecs]
#     f = [1-pdist(np.vstack((vec_model,vec)),metric=metric)[0] for vec in vecs]
#     return f


# def plot_rsa_results(rsa_results,ax=None,title=None,xlbls=['BG','SL','VAE']):
#     m = rsa_results.mean(axis=0)
#     se = rsa_results.std(axis=0)
#     xs = np.arange(len(m))
#     fmt = 'r '
#     if not ax:
#         plt.bar(xs,m);
#         plt.errorbar(xs,m,se,fmt=fmt);
#         plt.xticks(xs,labels=xlbls);
#     else:
#         ax.bar(xs,m);
#         ax.errorbar(xs,m,se,fmt=fmt);
#         ax.set_xticks(xs)
#         #ax.set_xticklabels(['BG','SL','VAE'][xs])
#         ax.set_xticklabels(np.array(xlbls)[xs])
#         ax.set_title(title)
        

# def plot_pca_rsa(keys,df,ABIDE_data,patients,encs,xlbls=['BG','SL','VAE'],thresh=.25):
#     '''Takes in an array of scores. Does PCA on them. Calculates RSA based on PCA. 
#     If PC explains morethan .25 of total variance - plots model fit and results'''
    
#     arr = df[keys].values
#     #isnan = np.isnan(arr[:,1]);
#     isnan = np.isnan(arr).sum(axis=1)!=0
#     print(f'{sum(patients[~isnan])} subjects')

#     n_components = 3
#     reducer = PCA(n_components=n_components)
#     components = reducer.fit_transform(arr[~isnan,:])
    
#     print(reducer.explained_variance_ratio_)

#     n_components = max(np.nonzero(reducer.explained_variance_ratio_>thresh)[0])+1

#     #rdms = [make_RDM(components[:,i]) for i in range(n_components)]
#     rdms = [make_RDM(components[patients[~isnan],i]) for i in range(n_components)]
#     f,ax = plt.subplots(1,3,figsize=(15,5))
#     #[sns.heatmap(rdms[i],cbar=[],ax=ax[i],xticklabels=[],yticklabels=[]) for i in range(n_components)]
#     [sns.heatmap(rdms[i],cbar=[],ax=ax[i]) for i in range(n_components)]


#     f,ax = plt.subplots(1,3,figsize=(15,5))
#     res = list()
#     for i in range(n_components):
#         rsa_results = [fit_rsa(components[:,i],ABIDE_data[~isnan,:,:,:],patients[~isnan],encs) for _ in range(10)]
#         rsa_results = np.array(rsa_results)
#         plot_rsa_results(rsa_results,ax=ax[i],xlbls=xlbls)
#         res.append(rsa_results)
        
#     return res


# def inverse_tx_umap(targ,s_embedding):
#     targ = np.array(targ)
#     idx = np.argsort(((s_embedding-targ)**2).sum(axis=1))
#     return idx


# def get_umap_corners(s_embedding):
#     ax_min = s_embedding.min(axis=0)
#     ax_max = s_embedding.max(axis=0)
#     ax_mid = s_embedding.mean(axis=0)
    
# #     ll = inverse_tx_umap((ax_min[0],ax_min[1]),s_embedding)
# #     lr = inverse_tx_umap((ax_max[0],ax_min[1]),s_embedding)
# #     tl = inverse_tx_umap((ax_min[0],ax_max[1]),s_embedding)
# #     tr = inverse_tx_umap((ax_max[0],ax_max[1]),s_embedding)

#     L = inverse_tx_umap((ax_min[0],ax_mid[1]),s_embedding)
#     R = inverse_tx_umap((ax_max[0],ax_mid[1]),s_embedding)
#     T = inverse_tx_umap((ax_mid[0],ax_max[1]),s_embedding)
#     B = inverse_tx_umap((ax_mid[0],ax_min[1]),s_embedding)
    
#     cntr = inverse_tx_umap(s_embedding.mean(axis=0),s_embedding)
    
#     map_ = dict()
# #     map_['ll'] = ll
# #     map_['lr'] = lr
# #     map_['tr'] = tr
# #     map_['tl'] = tl

#     map_['L'] = L
#     map_['R'] = R
#     map_['T'] = T
#     map_['B'] = B
#     map_['center'] = cntr

#     return map_