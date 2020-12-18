import numpy as np
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')

class KMeans_plus:
    def __init__(self, ImageNumber, K):
        self.ImageNumber = ImageNumber
        self.K = K

    def Kpp_setinit(self):
        pick = np.random.randint(0, len(self.total), 1)
        pick_points = self.total[pick]
        total_inside = self.total.copy()
        initial_points = np.zeros([self.K, 128])
        initial_points[0, :] = pick_points
        print("Setting Initial Points with K++ Algorithm")
        first=True
        for i in tqdm(range(self.K)):
            top = []
            pick_points = initial_points[:i+1, :]
            for pick_point in (pick_points):
                down = np.sum(np.square(total_inside - pick_point), axis = 1)
                top.append(down)
            top = np.array(top)
            if not first:     
                belong = np.min(top, axis = 0)
            else:
                belong = top[0]
                first = False
            max_idx = np.argmax(belong, axis = 0)
            pick_point = total_inside[max_idx]
            initial_points[i, :] = pick_point

        return initial_points

    def fit(self, total, iters = 10):
        """ K-means Algorithm수행하여 센터 탐색, trainset=sift_features iters=iteration(epochs) 
            OUTPUT = Centroids위치 (n x 128), 각 시프트 피처가 어디에 속해있는지(sklearn의 predict결과값)  """
        self.total = np.array(total)
        SIZEOFTOTAL = len(self.total)
        LOCAL_CENTROID = self.Kpp_setinit()
        
        for it in range(iters): # iteratoin 반복문
            print(f"ITERATION: {it}")
            BELONG = [0 for i in range(len(self.total))]
            ERROR_WHOLE = 0

            for idx, sift in enumerate(tqdm((self.total))):
                equation = np.sqrt(np.sum(np.square(LOCAL_CENTROID - sift), axis = 1))
                centriod_idx = np.argmin(equation)
                BELONG[idx] = centriod_idx
                ERROR_WHOLE += np.min(equation)

            BELONG = np.array(BELONG)
            print("Update CENTROID")
            for i in range(self.K):
                LOCAL_CENTROID[i, :] = np.mean(self.total[np.arange(SIZEOFTOTAL)[BELONG == i]], axis = 0)
                
        return LOCAL_CENTROID, BELONG

class VLAD:
    """ sift_list = 각 이미지의 시프트피쳐를 넣어준다. 
            total = 이미지별 시프트 피쳐를 이미지별 리스트로 묶지 않고 풀어서 [n x 128]의 넘파이 배열로 넣어준다.
            n_images = 이미지의 개수
            dim_limint = des를 몇 차원으로 생성할지 결정"""
    def __init__(self, sift_list, total, n_images, dim_limit):
        self.WHOLE = sift_list
        self.N_IMAGES = n_images
        self.DIM_LIMIT = dim_limit
        self.TOTAL = total

    def fit(self, centroids, belong):
        """ centroids = 중심점 개수 // belong = 각 sift feature가 어떤 클러스터에 속해있는지를 표현 
            A4_2018310412라는 이름으로 des파일을 생성해준다.
            numpy활용을 극대화하여 연산속도를 올렸습니다."""
        return_matrix = np.zeros((self.N_IMAGES, self.DIM_LIMIT), dtype="float32")
        index = 0
        for idx, sifts in enumerate(self.WHOLE):
            next_index = index + len(sifts)
            for cent in range(8): #centroids index
                belongs =  belong[index:next_index]
                residual = np.sum(sifts[belongs == cent] - centroids[cent], axis = 0)
                return_matrix[idx, cent*128:(cent+1)*128] = np.sign(residual) * np.log(np.abs(residual)+1) 
            index = next_index
            return_matrix[idx] = return_matrix[idx]/np.linalg.norm(return_matrix[idx], ord=2) #l2 normalizatiodn
            
        header1 = np.array([self.N_IMAGES,self.DIM_LIMIT],dtype=np.int32)
        print("Generating Descriptor File")
        with open("A4_2018310412.des", "wb") as f:
            f.write(header1.tobytes())
            f.write(return_matrix.tobytes())