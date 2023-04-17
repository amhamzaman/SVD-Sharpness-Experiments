import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





class SVD:
    def __init__(self) -> None:
        self.frame_size = 1000


    def variance(self, s):

        var = np.round(s**2/np.sum(s**2), decimals= 6)

        # print(sum(var))
        # print(var[:5])
        return var[:20]

    def compute_svd(self, img):
        
        u, s, v = np.linalg.svd(img)

        # print(s.shape)
        print(s[0:5])

        return u, s, v

    def preprocess(self, img):
        blurred = cv.GaussianBlur(img, ksize=(15, 15), sigmaX= 0.5, sigmaY= 0.5, borderType= cv.BORDER_DEFAULT)
        gradx = cv.Sobel(blurred, cv.CV_16S, 1, 0, ksize=3, scale= 1, delta=0, borderType=cv.BORDER_DEFAULT)
        grady = cv.Sobel(blurred, cv.CV_16S, 0, 1, ksize=3, scale= 1, delta=0, borderType=cv.BORDER_DEFAULT)
        abs_gradx = cv.convertScaleAbs(gradx)
        abs_grady = cv.convertScaleAbs(grady)
        grad = cv.addWeighted(abs_gradx, 0.5, abs_grady, 0.5, 0)

        mn = grad.min()
        mx = grad.max()

        gradn = (grad - mn)/(mx-mn)
        gradn = (gradn*255).astype('uint8')
        # print(gradn.max(), gradn.min())

        # can = cv.Canny(blurred, 10, 40)

        # cv.imshow('gradient', can)
        # cv.imshow('gradient_norm', gradn)
        # cv.imshow('sharpenned', cv.add(img, can))
        sharp = cv.add(img, grad)

        return blurred, sharp

    def reconstruct(self, nums, u, s, v):

        # low = u @ s @ v
        # low = u[:, :10] @ np.diag(s[:10]) @ v[:10, :]
        

        recons = []
        for i in range(len(nums)):
            low = u[:, :nums[i]] @ np.diag(s[:nums[i]]) @ v[:nums[i], :]
            # print(low.min(), low.max(), low.mean())
            low = cv.normalize(low, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            recons.append(low)
            # cv.imshow(f'reconstructed_{nums[i]}', low)
            # cv.imwrite(f"D:/Stacks/Reconstructed/recon_{nums[i]}.jpg", low)

        return recons

    def plot(self, var):
        
        x = [i for i in range(len(var))]

        plt.plot(x, var)
        plt.show()

    def plot_all(self, S):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        x = [i for i in range(len(S[0]))]
        for i,s in enumerate(S):
            plt.plot(x, s, colors[i], label= f"blur_{i}")

        plt.legend()

        plt.show()



    def get_frames(self, img):

        n = self.frame_size
        h, w = img.shape

        for i in range(0, h, n):
            for j in range(0, w, n):
                #print(f"[{j}:{j+n},{i}:{i+n}]")
                yield img[i : min(i+n,h), j : min(j+n,w)]

    def process_frame(self, frame):
        res, sharp = self.preprocess(frame)

        u, s, v = self.compute_svd(sharp)
        var = self.variance(s)
        # self.reconstruct(u, s, v)
        
        return s[:20], sharp
        

    def process_all(self, img):

        for i, frame in enumerate(self.get_frames(img)):
            if i > 0: break
            self.process_frame(frame)


    def blur_experiment(self, folder):
        S = []
        for root, _, files in os.walk(folder):
            for file in files:
                image_path = os.path.join(root, file)
                print(image_path)
                img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                s = self.process_frame(img)
                S.append(s)

        self.plot_all(S)

    def get_image(self, file_name):
        return cv.imread(file_name, cv.IMREAD_GRAYSCALE)

    def get_eigen_values(self, file_name):
        img = self.get_image(file_name)
        s, gradn = self.process_frame(img)
        return s, gradn

    def get_reconstructed(self, file_name, nums):
        img = self.get_image(file_name)
        _, sharp = self.preprocess(img)

        u, s, v = self.compute_svd(sharp)
        
        recons = self.reconstruct(nums, u, s, v)

        return recons, s

        

    def run(self, file_name):
        img = self.get_image(file_name)
        
        #self.get_frames(img)
        self.process_all(img)


def run_folder(folder):

    s = SVD()
    s.run(file_name)

    for root, _, files in os.walk(folder):
        for file in files:
            image_path = os.path.join(root, file)            
            eiv, gradn = s.get_eigen_values(image_path)
            gradn = cv.cvtColor(gradn, cv.COLOR_GRAY2BGR)
            
            img = cv.imread(image_path)
            res = cv.hconcat([img, gradn])

            imgt = cv.putText(res, f"{round(eiv[0], 2)}", (30,35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (250, 110, 150), 2, cv.LINE_AA)
            imgt = cv.putText(res, f"{round(sum(eiv[:6]), 2)}", (30,70), cv.FONT_HERSHEY_SIMPLEX, 1.0, (250, 110, 150), 2, cv.LINE_AA)
            cv.imwrite(f"D:/Stacks/DL Frames/Out_13100/{file}", imgt)


def run_reconstructor(folder):

    s = SVD()
    s.run(file_name)

    nums = [1, 5, 10, 20, 50, 100, 500, 1000]

    for root, _, files in os.walk(folder):
        for file in files:
            image_path = os.path.join(root, file)            
            recons, eig = s.get_reconstructed(image_path, nums)
            for i, rec in enumerate(recons):
                # print(rec.shape)
                cv.cvtColor(rec, cv.COLOR_GRAY2BGR)
                cv.putText(rec, f"{nums[i]}", (30,35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (250, 110, 150), 2, cv.LINE_AA)
                cv.putText(rec, f"{round(sum(eig[:nums[i]]), 2)}", (30,75), cv.FONT_HERSHEY_SIMPLEX, 1.0, (250, 110, 150), 2, cv.LINE_AA)
            
            r1 = cv.hconcat([recons[0], recons[1], recons[2], recons[3]])
            r2 = cv.hconcat([recons[4], recons[5], recons[6], recons[7]])
            res = cv.vconcat([r1, r2])
            
            cv.imwrite(f"D:/Stacks/DL Frames/Out_13100/{file}", res)


def create_df():
    folder_list = ["D:/Stacks/DL Frames/13100/", "D:/Stacks/DL Frames/011851/", "D:/Stacks/DL Frames/02987/"]
    folder_list = ["D:/Signal Detection Data/DAPI/DAPI/4/"]
    s = SVD()
    df = pd.DataFrame(columns=['Folder', 'Image', 'Eig_1', 'Eig_2', 'Eig_3', 'Eig_4', 'Eig_5'])

    for folder in folder_list:
        fol = folder.split('/')[-2]
        for root, _, files in os.walk(folder):
            
            for i, file in enumerate(files):
                image_path = os.path.join(root, file)            
                eig, _ = s.get_eigen_values(image_path)
                df.loc[len(df)] = [fol, i, eig[0], eig[1], eig[2], eig[3], eig[4]]
                img = cv.imread(image_path)
                cv.putText(img, f"{round(eig[0], 2)}", (3,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (250, 110, 150), 2, cv.LINE_AA)
                cv.imwrite(f"D:/Stacks/DL Frames/Out_DAPI_2/{i}.jpg", img)

                
    df.to_csv('D:/Stacks/DL Frames/eigenvals_DAPI.csv')


if __name__ == "__main__":
    
    
    file_name = "D:/Stacks/C224850-LH-D13-60X-05252022_STACKED/DAPI.jpg"
    folder = "D:/Stacks/Blurred/"
    folder = "D:/Stacks/DL Frames/13100/"
    # folder = "D:/Signal Detection Data/DAPI/DAPI/3/"

    # run_folder(folder)
    # run_reconstructor(folder)
    create_df()

    # s = SVD()
    # s.run(file_name)
    # s.blur_experiment(folder)


    cv.waitKey(0)
    cv.destroyAllWindows()