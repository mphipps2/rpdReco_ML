def get_dataset(folder = "./Data/reduced_tree_tmp/", side = 'A'):
        output = []
        start = time.time()

        data = pd.read_pickle(folder + f"{side}charge.pickle")
        return data
        # for i in range(100, 1000100, 100):
        #         if i % 100000 == 0:
        #             print("event", i, "time", time.time() - start)
        #             start = time.time()
        #         output.append(pd.read_pickle(folder + f"{side}_RPD_photon{i}.pickle"))
        # return pd.concat(output, ignore_index = True).set_index('Event number').astype(float)
