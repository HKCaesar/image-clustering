from scipy.spatial.distance import pdist,squareform,euclidean
from scipy.misc import imsave
from scipy.ndimage import imread
from shutil import copyfile, rmtree
from glob import glob
from time import time
from numpy import array,reshape,where,concatenate,flipud,double
from numpy.random import shuffle
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from gc import collect
from Pycluster import somcluster
from math import sqrt,floor
import rpy2.robjects.numpy2ri
import os
rpy2.robjects.numpy2ri.activate()
cba = importr("cba")
stats = importr("stats")
dtw = importr("dtw")

HTML_BEGINNING = '<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Cluster Analysis Methods Comparison</title><style>td{width:33%;margin:auto;}table{width:100%}img{width:100%}</style></head><body>'
HTML_END = '</body></html>'

def generate_results_as_html(convert_to_gray = False, shuffle_rows = True, shuffle_columns = True):
    if os.path.exists("Output"):
        rmtree("Output") 
    os.mkdir("Output")
    html = HTML_BEGINNING
    image_paths = glob("Images\\*.png")
    image_paths.extend(glob("Images\\*.jpg"))
    image_paths.extend(glob("Images\\*.GIF"))
    for image_path in image_paths:
        html += generate_html_for_image(image_path, convert_to_gray, shuffle_rows, shuffle_columns)
    html += HTML_END
    output = open("Output\\index.html", "w")
    output.write(html)
    output.close()

def generate_html_for_image(image_path, convert_to_gray, shuffle_rows, shuffle_columns):
    orig_image_name = image_path.split("\\")[1]
    copyfile(image_path, "Output\\" + orig_image_name)
    image_data = init_image_data(image_path, convert_to_gray)
    html = '<center><h1>' + orig_image_name + '</h1></center><table><tr><td><center><h2>Original</h2><h4>(Resolution: ' + str(len(image_data[0])) + 'x' + str(len(image_data)) + ' pixels)</h4></center><img src="' + orig_image_name + '"></td>'
    
    shuffle_image_data(image_data, shuffle_rows, shuffle_columns)
    imsave("Output\\shuffled_" + orig_image_name, image_data)
    html += '<td><center><h2>Shuffled</h2><h4>(Rows shuffled: ' + str(shuffle_rows) + ', Columns shuffled: ' + str(shuffle_columns) + ', Grayscale: ' + str(convert_to_gray) + ')</h4></center><img src="shuffled_' + orig_image_name + '"></td>'
    
    greedy_order_image = do_greedy_order(image_data, "euclidean", convert_to_gray, shuffle_rows, shuffle_columns)
    imsave("Output\\greedy_" + orig_image_name, greedy_order_image[0])
    html += '<td><center><h2>Greedy Ordering (' + str(greedy_order_image[1]) + 's)</h2><h4>(Similarity metric: ' + greedy_order_image[2] + ')</h4></center><img src="greedy_' + orig_image_name + '" width="100%"></td></tr>'
    
    hierarchical_image = do_hierarchical(image_data, "euclidean", "single", convert_to_gray, shuffle_rows, shuffle_columns)
    imsave("Output\\hierarchical_" + orig_image_name, hierarchical_image[0])
    html += '<tr><td><center><h2>Hierarchical Clustering (' + str(hierarchical_image[1]) + 's)</h2><h4>(Similarity metric: ' + hierarchical_image[3] + ', Clustering method: ' + hierarchical_image[2] + ')</h4></center><img src="hierarchical_' + orig_image_name + '"></td>' 
    
    hierarchical_optimal_image = do_hierarchical_optimal(image_data, "euclidean", "single", convert_to_gray, shuffle_rows, shuffle_columns)
    imsave("Output\\hierarchical_optimal_" + orig_image_name, hierarchical_optimal_image[0])
    html += '<td><center><h2>Optimal Hierarchical Clustering(' + str(hierarchical_optimal_image[1]) + 's)</h2><h4>(Similarity metric: ' + hierarchical_optimal_image[3] + ', Clustering method: ' + hierarchical_optimal_image[2] + ')</h4></center><img src="hierarchical_optimal_' + orig_image_name + '"></td>' 
    
    kmeans_image = do_kmeans(image_data, len(image_data) // 10, 10000, 1, convert_to_gray, shuffle_rows, shuffle_columns)
    imsave("Output\\kmeans_" + orig_image_name, kmeans_image[0])
    html += '<td><center><h2>K-Means Clustering(' + str(kmeans_image[1]) + 's)</h2><h4>(Similarity metric: ' + kmeans_image[5] + ', Clusters: ' + str(kmeans_image[2]) + ', Iterations: ' + str(kmeans_image[3]) + ', Starts: ' + str(kmeans_image[4]) + ')</h4></center><img src="kmeans_' + orig_image_name + '"></td></tr>' 
    
    kmeans_greedy_image = do_kmeans_greedy(image_data, len(image_data) // 10, 10000, 1, convert_to_gray, shuffle_rows, shuffle_columns)
    imsave("Output\\kmeans_greedy_" + orig_image_name, kmeans_greedy_image[0])
    html += '<tr><td><center><h2>Greedy K-Means Clustering(' + str(kmeans_greedy_image[1]) + 's)</h2><h4>(Similarity metric: ' + kmeans_greedy_image[5] + ', Clusters: ' + str(kmeans_greedy_image[2]) + ', Iterations: ' + str(kmeans_greedy_image[3]) + ', Starts: ' + str(kmeans_greedy_image[4]) + ')</h4></center><img src="kmeans_greedy_' + orig_image_name + '"></td>' 
   
    som_image = do_som(image_data, len(image_data) // 10, 75000, 0.9, convert_to_gray, shuffle_rows, shuffle_columns)
    imsave("Output\\som_" + orig_image_name, som_image[0])
    html += '<td><center><h2>1-D Self-Organizing Map(' + str(som_image[1]) + 's)</h2><h4>(Similarity metric: ' + som_image[2] + ', Clusters: ' + str(som_image[3]) + ', Iterations: ' + str(som_image[4]) + ', Tau: ' + str(som_image[5]) + ')</h4></center><img src="som_' + orig_image_name + '"></td>' 

    greedy_som_image = do_greedy_som(image_data, len(image_data) // 10, 75000, 0.9, convert_to_gray, shuffle_rows, shuffle_columns)
    imsave("Output\\greedy_som_" + orig_image_name, greedy_som_image[0])
    html += '<td><center><h2>Greedy 1-D Self-Organizing Map(' + str(greedy_som_image[1]) + 's)</h2><h4>(Similarity metric: ' + greedy_som_image[2] + ', Clusters: ' + str(greedy_som_image[3]) + ', Iterations: ' + str(greedy_som_image[4]) + ', Tau: ' + str(greedy_som_image[5]) +  ')</h4></center><img src="greedy_som_' + orig_image_name + '"></td></tr></table>' 

    return html

def init_image_data(image_file, convert_to_gray):
    if convert_to_gray:
        return imread(image_file, flatten = True).astype(double)
    return imread(image_file, mode = "RGB").astype(double)

def shuffle_image_data(image_data, shuffle_rows, shuffle_columns):
    if shuffle_rows:
        shuffle(image_data)
    if shuffle_columns:
        shuffle(image_data.swapaxes(0,1))

#Possible similarity metrics: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html + custom methods, currently "DTW"
#Possible clustering methods: "ward.D", "ward.D2", "single", "complete", "average" (= UPGMA), "mcquitty" (= WPGMA), "median" (= WPGMC) or "centroid" (= UPGMC)
def do_hierarchical_optimal(data, metric, method, convert_to_gray, shuffle_rows, shuffle_columns):
    time_start = time()
    result = data
    if shuffle_rows:
        result = reshape(result, (len(result), result[0].size))
        result = calc_hierarchical_optimal(result, metric, method)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
    if shuffle_columns:
        result = result.swapaxes(0, 1)
        result = reshape(result, (len(result), result[0].size))
        result = calc_hierarchical_optimal(result, metric, method)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
        result = result.swapaxes(0, 1)
    time_total = round(time() - time_start, 3)
    return (result, time_total, method, metric)

def calc_hierarchical_optimal(data, metric, method):
    if metric == "dtw":
        r_dist = r.dist(data, method = "DTW", window_type = "sakoechiba", window_size = len(data) // 50, distance_only ='TRUE')
    else:
        dist_matrix = squareform(pdist(data, metric))
        r_dist = stats.as_dist(dist_matrix)
    merge_data = array(r.hclust(r_dist, method = method)[0])
    ordered_data = cba.order_optimal(r_dist, merge_data)[1]
    indexes = array(ordered_data)
    reordered_data = []
    for index in indexes:
        reordered_data.append(data[index - 1])
    return array(reordered_data)
    

#Possible similarity metrics: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html + custom methods, currently "DTW"
#Possible clustering methods: "ward.D", "ward.D2", "single", "complete", "average" (= UPGMA), "mcquitty" (= WPGMA), "median" (= WPGMC) or "centroid" (= UPGMC)
def do_hierarchical(data, metric, method, convert_to_gray, shuffle_rows, shuffle_columns):
    time_start = time()
    result = data
    if shuffle_rows:
        result = reshape(result, (len(result), result[0].size))
        result = calc_hierarchical(result, metric, method)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
    if shuffle_columns:
        result = result.swapaxes(0, 1)
        result = reshape(result, (len(result), result[0].size))
        result = calc_hierarchical(result, metric, method)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
        result = result.swapaxes(0, 1)
    time_total = round(time() - time_start, 3)
    return (result, time_total, method, metric)

def calc_hierarchical(data, metric, method):
    if metric == "dtw":
        r_dist = r.dist(data, method = "DTW", window_type = "sakoechiba", window_size = len(data) // 50, distance_only ='TRUE')
    else:
        dist_matrix = squareform(pdist(data, metric))
        r_dist = stats.as_dist(dist_matrix)
    indexes = array(r.hclust(r_dist, method = method)[2])
    reordered_data = []
    for index in indexes:
        reordered_data.append(data[index - 1])
    del indexes
    return array(reordered_data)

#Possible similarity metrics: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html + custom methods, currently "DTW"
def do_greedy_order(data, metric, convert_to_gray, shuffle_rows, shuffle_columns):
    time_start = time()
    result = data
    if shuffle_rows:
        result = reshape(result, (len(result), result[0].size))
        result = calc_greedy_order(result, metric)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
    if shuffle_columns:
        result = result.swapaxes(0, 1)
        result = reshape(result, (len(result), result[0].size))
        result = calc_greedy_order(result, metric)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
        result = result.swapaxes(0, 1)
    time_total = round(time() - time_start, 3)
    return (result, time_total, metric)

def calc_greedy_order(data, metric):
    if metric == "dtw":
        r_dist = r.dist(data, method = "DTW", window_type = "sakoechiba", window_size = len(data) // 50, distance_only ='TRUE') 
    else:
        dist_matrix = squareform(pdist(data, metric))
        r_dist = stats.as_dist(dist_matrix) 
    ordered_data = cba.order_greedy(r_dist)[1]
    indexes = array(ordered_data)
    reordered_data = []
    for index in indexes:
        reordered_data.append(data[index - 1])
    return array(reordered_data)

#Possible similarity metrics: "euclidean"
def do_kmeans(data, clusters, iterations, starts, convert_to_gray, shuffle_rows, shuffle_columns):
    time_start = time()
    result = data
    if shuffle_rows:
        result = reshape(result, (len(result), result[0].size))
        result = calc_kmeans(result, clusters, iterations, starts)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
    if shuffle_columns:
        result = result.swapaxes(0, 1)
        result = reshape(result, (len(result), result[0].size))
        result = calc_kmeans(result, clusters, iterations, starts)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
        result = result.swapaxes(0, 1)
    time_total = round(time() - time_start, 3)
    return (result, time_total, clusters, iterations, starts, "euclidean")

def calc_kmeans(data, clusters, iterations, starts):
    data_as_list = data.flatten()
    data_as_rmatrix = r.matrix(data_as_list, ncol = len(data[0]), byrow = True)
    cluster_indexes = array(r.kmeans(data_as_rmatrix, clusters, iter_max = iterations, nstart = starts)[0])
    reordered_data = []
    for i in range (1, clusters + 1):
        indexes = where(cluster_indexes == i)[0]
        for index in indexes:
            reordered_data.append(data[index])        
    return array(reordered_data)

#Possible similarity metrics: "euclidean"
def do_kmeans_greedy(data, clusters, iterations, starts, convert_to_gray, shuffle_rows, shuffle_columns):
    time_start = time()
    result = data
    if shuffle_rows:
        result = reshape(result, (len(result), result[0].size))
        result = calc_kmeans_greedy(result, clusters, iterations, starts)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
    if shuffle_columns:
        result = result.swapaxes(0, 1)
        result = reshape(result, (len(result), result[0].size))
        result = calc_kmeans_greedy(result, clusters, iterations, starts)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
        result = result.swapaxes(0, 1)
    time_total = round(time() - time_start, 3)
    return (result, time_total, clusters, iterations, starts, "euclidean")

def calc_kmeans_greedy(data, clusters, iterations, starts):
    data_as_list = data.flatten()
    data_as_rmatrix = r.matrix(data_as_list, ncol = len(data[0]), byrow = True)
    cluster_indexes = array(r.kmeans(data_as_rmatrix, clusters, iter_max = iterations, nstart = starts)[0])
    ordered_clusters = order_kcluster_contents(clusters, cluster_indexes, data)
    return order_clusters(ordered_clusters)    

def order_kcluster_contents(clusters, cluster_indexes, data):
    ordered_clusters = []
    for i in range (1, clusters + 1):
        cluster_data = []
        indexes = where(cluster_indexes == i)[0]
        for index in indexes:
            cluster_data.append(data[index])
        ordered_clusters.append(calc_greedy_order(array(cluster_data), "euclidean"))
    return ordered_clusters

def order_clusters(ordered_clusters):
    reordered_data = []
    reordered_data.append(ordered_clusters[0])
    used_indexes = [0]
    while len(used_indexes) < len(ordered_clusters):                      
        nearest_start = find_nearest(reordered_data[0][0], ordered_clusters, used_indexes)
        nearest_end = find_nearest(reordered_data[-1][-1], ordered_clusters, used_indexes)
        if nearest_start[0] < nearest_end[0]:
            if nearest_start[2] == 'end':
                reordered_data.insert(0, ordered_clusters[nearest_start[1]])
            else:
                reordered_data.insert(0, flipud(ordered_clusters[nearest_start[1]]))
            used_indexes.append(nearest_start[1])
        else:
            if nearest_end[2] == 'end':
                reordered_data.append(flipud(ordered_clusters[nearest_end[1]]))
            else:
                reordered_data.append(ordered_clusters[nearest_end[1]])
            used_indexes.append(nearest_end[1])
    return concatenate(reordered_data)

def find_nearest(row_1, clusters, used_indexes):
    closest = (float('inf'), None, None)
    for i in range (len(clusters)):
        if i in used_indexes:
            continue
        dist_start = euclidean(row_1, clusters[i][0])
        dist_end = euclidean(row_1, clusters[i][-1])
        if dist_start < closest[0]:
            closest = (dist_start, i, 'start')
        if dist_end < closest[0]:
            closest = (dist_end, i, 'end')
    return closest

#Possible similarity metrics: "euclidean"
def do_som(data, clusters, iterations, tau, convert_to_gray, shuffle_rows, shuffle_columns):
    time_start = time()
    result = data
    if shuffle_rows:
        result = reshape(result, (len(result), result[0].size))
        result = calc_som(result, clusters, iterations, tau)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
    if shuffle_columns:
        result = result.swapaxes(0, 1)
        result = reshape(result, (len(result), result[0].size))
        result = calc_som(result, clusters, iterations, tau)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
        result = result.swapaxes(0, 1)
    time_total = round(time() - time_start, 3)
    return (result, time_total, "euclidean", clusters, iterations, tau)

def calc_som(data, clusters, iterations, tau):
    time_start = time()
    cluster_indexes, celldata = somcluster(data, mask=None, weight=None, transpose=0, nxgrid=clusters, nygrid=1, inittau=tau, niter=iterations, dist='e')
    ordered_clusters = []
    for i in range (clusters):
        cluster_data = []
        indexes = []
        for j in range (len(cluster_indexes)):
            if cluster_indexes[j][0] == i:
                indexes.append(j)
        for index in indexes:
            cluster_data.append(data[index])
        if indexes == []:
            continue
        ordered_clusters.append(array(cluster_data))
    return concatenate(ordered_clusters)

def do_greedy_som(data, clusters, iterations, tau, convert_to_gray, shuffle_rows, shuffle_columns):
    time_start = time()
    result = data
    if shuffle_rows:
        result = reshape(result, (len(result), result[0].size))
        result = calc_greedy_som(result, clusters, iterations, tau)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
    if shuffle_columns:
        result = result.swapaxes(0, 1)
        result = reshape(result, (len(result), result[0].size))
        result = calc_greedy_som(result, clusters, iterations, tau)
        if not convert_to_gray:
            result = reshape(result, (len(result), result[0].size // 3, 3))
        result = result.swapaxes(0, 1)
    time_total = round(time() - time_start, 3)
    return (result, time_total, "euclidean", clusters, iterations, tau)

def calc_greedy_som(data, clusters, iterations, tau):
    cluster_indexes, celldata = somcluster(data, mask=None, weight=None, transpose=0, nxgrid=clusters, nygrid=1, inittau=tau, niter=iterations, dist='e')
    ordered_clusters = []
    for i in range (clusters):
        cluster_data = []
        indexes = []
        for j in range (len(cluster_indexes)):
            if cluster_indexes[j][0] == i:
                indexes.append(j)
        for index in indexes:
            cluster_data.append(data[index])
        if indexes == []:
            continue
        ordered_cluster = calc_greedy_order(array(cluster_data), "euclidean")
        if ordered_clusters == []:
            ordered_clusters.append(ordered_cluster)
            continue
        if len(ordered_clusters) == 1:
            last_first = euclidean(ordered_clusters[0][-1], ordered_cluster[0])
            last_last = euclidean(ordered_clusters[0][-1], ordered_cluster[-1])
            first_last = euclidean(ordered_clusters[0][0], ordered_cluster[-1])
            first_first = euclidean(ordered_clusters[0][0], ordered_cluster[0])
            min_dist = [last_first, last_last, first_last, first_first]
            if min_dist == last_first:
                ordered_clusters.append(ordered_cluster)
            elif min_dist == last_last:
                ordered_clusters.append(flipud(ordered_cluster))
            elif min_dist == first_first:
                ordered_clusters[0] = flipud(ordered_clusters[0])
                ordered_clusters.append(ordered_cluster)
            else:
                ordered_clusters[0] = flipud(ordered_clusters[0])
                ordered_clusters.append(flipud(ordered_cluster))              
            continue
        if (euclidean(ordered_clusters[-1][-1], ordered_cluster[0]) < euclidean(ordered_clusters[-1][-1], ordered_cluster[-1])):
            ordered_clusters.append(ordered_cluster)
        else:
            ordered_clusters.append(flipud(ordered_cluster))        
    return concatenate(ordered_clusters)

t = time()
generate_results_as_html(shuffle_columns=False)
print (round(time() - t, 3), 'sec elapsed')
