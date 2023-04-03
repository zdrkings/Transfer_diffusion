from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, img_data, eta_data, idx_data):

        self.img_data = img_data
        self.eta_data = eta_data
        self.idx_data = idx_data

    def __getitem__(self, index):

        piece_img_data = self.img_data[index]
        piece_eta_data = self.eta_data[index]
        piece_idx_data = self.idx_data[index]

        return  piece_img_data, piece_eta_data, piece_idx_data

    def __len__(self):
        return len(self.img_data)

