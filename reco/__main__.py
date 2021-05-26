from reco.reco import input_data
from reco.reco.train_cnn import train_cnn
from reco.reco.test_cnn import test_cnn

if __name__ == '__main__':

        # Run Mode:
        # Get Input Data: 0
        # Train: 1
        # Test: 2
        run_mode = 1
        
        # Model options:
        # Center-of-mass: 0
        # CNN: 1
        # Linear: 2
        # FC: 3
        # XGBoost: 4
        # Ensemble: 5
        model_choice = 1

        if run_mode == 0:
                input_data.get_reco_input()                
        elif run_mode == 1: 
                if model_choice == 0:
                        print('Center-of-mass does not need trained!')
                elif model_choice == 1:
                        print('Training CNN model')
                        train_cnn.run()
                elif model_choice == 2:
                        print('Training linear model')
                        train_linear.run()
        elif run_mode == 2: 
                if model_choice == 0:
                        print('Center-of-mass does not need trained!')
                elif model_choice == 1:
                        print('Training CNN model')
                        train_cnn.run()
                elif model_choice == 2:
                        print('Training linear model')
                        train_linear.run()
