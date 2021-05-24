def norm_neutron(n_hit):
                
        print('unnormalized n_hit: ' , n_hit)
        n_hit -= n_hit.mean(axis=0)
        print('normalized mean: ' , n_hit)
        n_hit /= n_hit.std(axis=0)
        print('normalized width: ' , n_hit)
        return n_hit

def z_norm(data):

        data['channel_15'] = data['channel_15'] - data['channel_15'].mean()
        data['channel_15'] = data['channel_15'] / data['channel_15'].std()

        data['channel_14'] = data['channel_14'] - data['channel_14'].mean()
        data['channel_14'] = data['channel_14'] / data['channel_14'].std()

        data['channel_13'] = data['channel_13'] - data['channel_13'].mean()
        data['channel_13'] = data['channel_13'] / data['channel_13'].std()

        data['channel_12'] = data['channel_12'] - data['channel_12'].mean()
        data['channel_12'] = data['channel_12'] / data['channel_12'].std()

        data['channel_11'] = data['channel_11'] - data['channel_11'].mean()
        data['channel_11'] = data['channel_11'] / data['channel_11'].std()

        data['channel_10'] = data['channel_10'] - data['channel_10'].mean()
        data['channel_10'] = data['channel_10'] / data['channel_10'].std()

        data['channel_9'] = data['channel_9'] - data['channel_9'].mean()
        data['channel_9'] = data['channel_9'] / data['channel_9'].std()

        data['channel_8'] = data['channel_8'] - data['channel_8'].mean()
        data['channel_8'] = data['channel_8'] / data['channel_8'].std()

        data['channel_7'] = data['channel_7'] - data['channel_7'].mean()
        data['channel_7'] = data['channel_7'] / data['channel_7'].std()

        data['channel_6'] = data['channel_6'] - data['channel_6'].mean()
        data['channel_6'] = data['channel_6'] / data['channel_6'].std()

        data['channel_5'] = data['channel_5'] - data['channel_5'].mean()
        data['channel_5'] = data['channel_5'] / data['channel_5'].std()

        data['channel_4'] = data['channel_4'] - data['channel_4'].mean()
        data['channel_4'] = data['channel_4'] / data['channel_4'].std()

        data['numParticle'] = data['numParticle'] - data['numParticle'].mean()
        data['numParticle'] = data['numParticle'] / data['numParticle'].std()
        return data

def get_unit_vector(arr):
        output = []
        norm =  np.linalg.norm(arr, axis = 1)
        for i in range(arr.shape[0]):
                output.append([arr[i][0] / norm[i], arr[i][1] / norm[i]])
        return np.array(output)

def average_vector(QA, QB):
    NormA = np.linalg.norm(QA, axis = 1)
    NormB = np.linalg.norm(QB, axis = 1)
    NA = np.array([QA[:,0 ] / NormA, QA[:,1 ] / NormA])
    NB = np.array([QB[:,0 ] / NormB, QB[:,1 ] / NormB])
    flip_B = -NB
    avgx = (NA[0] + flip_B[0]) / 2     
    avgy = (NA[1] + flip_B[1]) / 2     
    avg = np.arctan2(avgy, avgx)    
    return NA, NB, avgx, avgy, avg

def process_signal(ary, normalization = False, flatten = False, pad = 1):
    if normalization:
            ary = np.array([i.reshape(4,4,1)/np.max(i) for i in ary])
    else:
        ary = np.array([i.reshape(4,4,1) for i in ary])
    if flatten:
            ary = np.array([i.reshape(16) for i in ary])
    if pad:
        ary = np.pad(ary[:, :, :, :], ((0, 0), (pad, pad), (pad, pad), (0,0)), 'constant')
    return ary

    
