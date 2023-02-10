import pandas as pd
from scipy.io import loadmat


def count_list_elements(lst):
    count = 0
    for item in lst:
        if isinstance(item, list):
            count += count_list_elements(item)
        else:
            count += 1
    return count


class SPMfile:

    def __init__(self,filepath):
        self.filepath = filepath
        self.spmmat = self.load_spm()
        self.design_matrix = self.load_spm_dm()

    def load_spm(self):
        return loadmat(self.filepath, struct_as_record=False)

    def load_spm_dm(self):
        designMatrix = self.spmmat['SPM'][0][0].xX[0][0].X
        names = [i[0] for i in self.spmmat['SPM'][0][0].xX[0][0].name[0]]
        design_matrix = pd.DataFrame(designMatrix,columns=names)
        return design_matrix

    def get_reg_index(self,target_name):
        target_index = []
        for i,reg_name in enumerate(self.design_matrix.columns):
            if target_name in reg_name:
                target_index.append(i+1)
        if len(target_index) == 0:
            print("The {} don't have regressor.".format(target_name))
        return target_index

    def get_regs_index(self,target_names):
        # reg_names: a list of regressor names
        targets_index = []
        for target_name in target_names:
            target_index = self.get_reg_index(target_name)
            targets_index.append(target_index)
        return targets_index

    def get_dof(self):
        df_full = self.design_matrix.shape[0] - self.design_matrix.shape[1]  # get the degrees of freedom of full model n-k^*
        return df_full


