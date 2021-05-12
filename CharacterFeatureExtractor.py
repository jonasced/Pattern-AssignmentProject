import numpy as np

### Function used in feature extractor

def removeZero(data):
# Removes all data points where b = 0 from data array
#
# Input:    data                from DrawCharacter (3,L) 
#
# Output    clean_data          (2,L-num_zero)

    b = data[2,:]
    index_ones = b==1
    x = data[0,:][index_ones]
    y = data[1,:][index_ones]
    clean_data = np.array([x, y])
    return clean_data


def sampler(coordinates,thr):
# Removes all samples for which the distance from the previous point is less than a given threshold
#
# Input:    coordinates         clean data from DrawCharacter with only x,y (2,L)
#           thr                 lower threshold bound for distances when removing samples
#
# Output:   sampled_coordinates sampled data (2,L-removed_samples)      

    rel_dist = np.diff(coordinates)
    #calculating absolute distances between each sample
    abs_dist = np.sqrt(    np.square(rel_dist[0,:]) + np.square(rel_dist[1,:])    )
    #vector for indices of output samples. Starting with keeping the first sample
    remain_sample_indices = [0]
    #keep the samples where their distances between each other is higher than the given threshold
    temp_dist = 0
    for i in range(abs_dist.size):
        temp_dist = temp_dist + abs_dist[i] 
        if temp_dist > thr:
            temp_dist = 0
            remain_sample_indices.append(i + 1)
    #getting x,y values of the samples for corresponding indices
    sampled_coordinates = coordinates[:,remain_sample_indices]
    return sampled_coordinates


def norm_dist_slope(coordinates,thr):
# Calculates a metric for scale of the symbol which is used to normalize large distances between two samples.
# Adds the large distance to the remaining values.
# Calculates the polar coordinates (absolute distance and slope) of the input coordinates and adds random noise to avoid zero variance.
#  
# 
# Input:    coordinates         sampled data (2,L-removed_samples)
#           thr                 threshold used to downsample, used to calculate "large" distances
# 
# Output:   dist_slope          polar coordinates with distance increased at each large distance increment

    #Calculation of maximum distance in the symbol to get an idea about scale of it
    init_coor = coordinates[:,0]
    init_coor = init_coor.reshape((init_coor.size,1))
    origin_coor = coordinates-init_coor
    distances = np.sqrt(    np.square(origin_coor[0,:]) + np.square(origin_coor[1,:])    )
    max_dist = np.max(distances)
    
    #Finding the distance values between strokes ending and starting far from each other and normalizing them to make them similar for
    #different scaled similar symbols
    rel_dist = np.diff(coordinates)
    abs_dist = np.sqrt(    np.square(rel_dist[0,:]) + np.square(rel_dist[1,:])    )
    #if absolute distance between two points is higher than 2 times threshold ( which means there is certainly a jump between strokes) then normalize that distance value
    bigjump_indices = np.where(abs_dist > 2*thr)
    abs_dist[bigjump_indices] = (abs_dist[bigjump_indices] / max_dist) * 5 * thr
    
    #indices of distance values where we have big jumps between strokes
    bigjump_array = bigjump_indices[0]
    
    #when we have a jump in distance value, we keep that in the same level for remaining distance values coming after that and we apply it for each jumps
    for jump in bigjump_array:
        shift = abs_dist[jump] - abs_dist[jump - 1]
        remaining = jump + 1
        abs_dist[remaining:] += shift
        
    #setting data type to float for both distance vectors
    abs_dist = abs_dist.astype(float)
    rel_dist = rel_dist.astype(float)
    
    #we calculate the degrees for each slope between consecutive samples. For cases x = 0 we set x to 0.0001 so that y/x will bi divisible
    zerox_indices = np.where(rel_dist[0,:] == 0)
    rel_dist[0,zerox_indices] = 0.0001
    slope = np.divide(rel_dist[1,:] , rel_dist[0,:] )
    slope_inf_indices = np.where(np.absolute(slope) > 1000)
    slope_angle = np.arctan(slope)
    #we set the corresponding angle of the slopes where we were supposed to have infinite to 90 degrees
    slope_angle[slope_inf_indices] = np.deg2rad(90)
    
    #reshaping arrays for concatenating properly
    abs_dist = np.reshape(abs_dist,(1,abs_dist.size))
    slope_angle = np.reshape(slope_angle,(1,slope_angle.size))

    # converts radian values to degrees
    slope_angle = np.rad2deg(slope_angle)
    
    #adding random noise for not having 0 variance in features
    abs_dist = abs_dist + np.random.rand(1)/100 
    slope_angle = slope_angle + np.random.rand(1)/100
    #returns feature matrix of size (2 x number of samples -1)
    dist_slope = np.concatenate((abs_dist,slope_angle),axis=0)

    return dist_slope


### Final feature extractor

def featureExtractor(symbol,thr, input_is_dc = True):
# Extracts feature vector from object of DrawCharacter
#
# Input:    symbol              object of DrawCharacter containing a drawn character
#           thr                 threshold for sampler
#
# Output:   feature_symbol      feature vector for the given drawn character (2,L-removed_samples)
#           sampled_symbol      xy-coordinates for character after sampling (2,L-removed_samples)

    if input_is_dc:
        xyb_values = symbol.get_xybpoints()
        xyb_cleaned = removeZero(xyb_values)
        sampled_symbol = sampler(xyb_cleaned, thr)
        feature_symbol = norm_dist_slope(sampled_symbol,thr)
    else:
        xyb_cleaned = removeZero(symbol)
        sampled_symbol = sampler(xyb_cleaned, thr)
        feature_symbol = norm_dist_slope(sampled_symbol,thr)
 
    return feature_symbol, sampled_symbol
