
def blur_neutron(n_hit):
        blur = []        
        for i in range(n_hit.shape[0]):
                blur.append(np.random.normal(n_hit[i], np.sqrt(n_hit[i] * (0.171702**2))))
        return np.array(blur)


def subtract_signals2(data):
        for x in range(15,3,-1):
                subtr_chan = x - 4
                data[f'channel_{x}'] = data[f'channel_{x}'] - data[f'channel_{subtr_chan}']
        return data


def subtract_signals3(data):
        for row in range(3,0,-1):
                for col in range(0,4,1):
                        subtr_row = row - 1
                        data[f'rpd{row}_{col}_Charge'] = data[f'rpd{row}_{col}_Charge'] - data[f'rpd{subtr_row}_{col}_Charge']
        return data


def process_signal(ary, normalization = False, flatten = False, padding = 1):

#        print('ary before: ', ary)
        if normalization:
#                print('signal before norm: ',ary)
                ary -= ary.mean(axis=(0,1))
                ary /= ary.std(axis=(0,1))
#                print('signal after norm: ',ary)
                ary = np.array([i.reshape(4,4,1) for i in ary])
        else:
                ary = np.array([i.reshape(4,4,1) for i in ary])
        if flatten:
                ary = np.array([i.reshape(16) for i in ary])
        if padding:
                ary = np.pad(ary[:, :, :, :], ((0, 0), (padding, padding), (padding, padding), (0,0)), 'constant')
#        print('ary after: ', ary)
        return ary

