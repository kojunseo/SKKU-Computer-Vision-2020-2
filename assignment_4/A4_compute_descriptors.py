from glob import glob
import numpy as np
from tqdm import tqdm
from Clustering import VLAD, KMeans_plus
from sklearn.cluster import KMeans

sift_list = []
for sift_path in sorted(glob("./sift/*"), key=lambda x:int(x[-4:])):
    with open(sift_path, "rb") as f:
        feature = np.fromfile(f, dtype=np.uint8).reshape(-1,128)
    sift_list.append(feature)
total = []
for idx, sifts in enumerate(tqdm(sift_list)):
    for sift in sifts:
        total.append(np.array(sift))
total = np.array(total)

#라이브러리를 써도 된다고 하셔서 쓴 버전입니다.
#속도를 위해서 미리 돌려놓고 numpy로 저장하여 불러와 사용해주었습니다.
print("Calculating K-means++ with sklearn library")
Km = KMeans(n_clusters=8, init="k-means++", random_state = 0).fit(total)
np.save("predict",Km.predict(total))
np.save("cluster_centers", Km.cluster_centers_)
cluster_centers = np.load("cluster_centers.npy") # 미리 돌려놓은 값 불러와줌
predict = np.load("predict.npy") # 미리 돌려놓은 값 불러와줌


#이건 직접짠건데, sklearn library써도 된다고 하셔서 위에 쓴 것입니다.
#아래 주석풀고 돌려주셔도 돌아갑니다! 성능차이도 거의 없습니다. 0.0x차이
# print("Calculating K-means++ with My own K-means Function")
# KM = KMeans_plus(1000, 8)
# cluster_centers, predict = KM.fit(total, iters = 10)

vlad = VLAD(sift_list, total, 1000, 1024)
vlad.fit(cluster_centers, predict)

