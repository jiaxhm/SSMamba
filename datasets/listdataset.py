import cv2
import numpy as np
import torch.utils.data as data



class ListDataset(data.Dataset):
    def __init__(self, root, dataset, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=None, datatype=None, scale=True):

        self.root = root
        self.dataset = dataset
        self.img_path_list =path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.datatype = datatype
        self.scale = scale

    def __getitem__(self, index):
        img_path = self.img_path_list[index][:-1]
        # We do not consider other datsets in this work
        assert self.dataset == 'bsd500'
        assert (self.transform is not None) and (self.target_transform is not None)

        inputs, label = self.loader(img_path, img_path.replace('_img.jpg', '_label.png'))
        l_edge = cv2.imread(img_path.replace('_img.jpg', '_edge.png'))


        l_edge = l_edge.astype(np.float32) / 255.
        l_edge[l_edge > 0.5] = 1.
        l_edge[(l_edge > 0) & (l_edge <= 0.5)] = 2.

        if self.scale:
            inputs, label, l_edge = self.generate_scale_label(inputs,label, l_edge.astype(np.uint8))
            

        if self.co_transform is not None:
            inputs, label = self.co_transform([inputs.astype(np.float32), l_edge], label)

        if self.transform is not None:
            image = self.transform(inputs[0])
        
        if self.target_transform is not None:
            #print(label.shape)
            label = self.target_transform(label[:,:,:1])
            l_edge = self.target_transform(inputs[1][:,:,:1])

        return image, label, l_edge

    def generate_scale_label(self, image, label, edge):
        f_scale = 0.7 + np.random.randint(0, 8) / 10.0
        image = cv2.resize(image, None,fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None,fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        edge = cv2.resize(edge, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, edge
        #return image, label

    def __len__(self):
        return len(self.img_path_list)
