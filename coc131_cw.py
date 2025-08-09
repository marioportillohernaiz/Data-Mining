from itertools import product
import os
import numpy
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats

# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.

optimal_hyperparam = {
    'hidden_layer_sizes': [(512, 128)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1.0],
    'learning_rate_init': [0.0001],
    'max_iter': [100],
}

class COC131:

    # Processing the images - resizing and then flattening
    def process_image(self, filepath, class_name):
        try:
            if not os.path.exists(filepath):
                # Finding the file with an extension
                directory = os.path.dirname(filepath)
                filename_base = os.path.basename(filepath)
                found = False
                
                if os.path.exists(directory):
                    for file in os.listdir(directory):
                        if file.startswith(filename_base + '.') or file == filename_base:
                            filepath = os.path.join(directory, file)
                            found = True
                            break
                
                if not found:
                    print(f"Could not find file: {filepath}")
                    return None, None
            
            img = Image.open(filepath)

            # Resizing to lower resolution
            img = img.resize((32, 32))
            img_array = numpy.array(img)

            if len(img_array.shape) > 2:
                img_array = numpy.mean(img_array, axis=2)
            flattened = img_array.flatten()

            return flattened, class_name
        
        except Exception as e:
            print(str(e))
            return None, None
        
    # Generating classification metrics 
    def get_classification_metrics(self, y_true, y_pred):
        class_names = [f"Class {int(i)}" for i in numpy.unique(y_true)]
        report = classification_report(y_true, y_pred, output_dict=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }

    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """

        # Setting dataset directory path
        dataset_path = "../dataset/"

        # Extracting class names and sorting them 
        class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        class_names.sort()
        class_to_label = {cls: float(i) for i, cls in enumerate(class_names)}

        files = [(os.path.join(dataset_path, cls, file), cls)
                for cls in class_names
                for file in os.listdir(os.path.join(dataset_path, cls))
                if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Processing all images
        processed = list(map(lambda x: self.process_image(x[0], x[1]), files))

        # Filtering out any failures and then unziping the data
        valid = [p for p in processed if p[0] is not None]
        data, labels = zip(*valid) if valid else ([], [])

        self.x = numpy.array(data, dtype=float)
        self.y = numpy.array([class_to_label[label] for label in labels], dtype=float)

        res1 = numpy.zeros(1)
        res2 = ''
        
        # Handling the specific file request if provided
        if filename:
            parts = filename.split('/')
            if len(parts) >= 2:
                file_class = parts[1]
                file_path = filename
                img_array, _ = self.process_image(file_path, file_class)

                if img_array is not None:
                    res1 = img_array
                    res2 = file_class
                else:
                    print(f"Error: Could not process file {filename}")
            else:
                print(f"Error: Invalid file path format {filename}")
        
        return res1, res2

    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """
        # Creating a StandardScaler which will then be used to standardize the data
        scaler = StandardScaler(with_mean=True, with_std=True)
        standardised_data = scaler.fit_transform(inp)
        
        # Scaling the standardised data to have std=2.5
        res2 = standardised_data * 2.5
        res1 = scaler
        
        return res2, res1

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """

        # Fetching standardised data using q2
        standardised_data, _ = self.q2(self.x)

        # Setting default test size if not provided
        if test_size is None:
            test_size = 0.3

        if pre_split_data is None:
            X_train, X_test, y_train, y_test = train_test_split(standardised_data, self.y, test_size=test_size, random_state=42, stratify=self.y)
        else:
            X_train, X_test, y_train, y_test = pre_split_data
        
        if hyperparam is None:
            hyperparam = optimal_hyperparam
        else:
            # Trainning the model with the hyperparams passed through
            best_model = None
            best_score = -1
            best_params = None
            param_keys = list(hyperparam.keys())
            param_values = list(hyperparam.values())
            param_combinations = list(product(*param_values))
            
            for combination in param_combinations:
                current_params = dict(zip(param_keys, combination))
                model = MLPClassifier(random_state=42, **current_params)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = current_params
            
            self.optimal_hyperparam = best_params

        params = {}
        for key, value in hyperparam.items():
            if isinstance(value, list) and len(value) == 1:
                params[key] = value[0]
            else:
                params[key] = value
        
        # Trainning the model with the most optimal hyperparams
        best_model = MLPClassifier(random_state=42, **params)
        best_model.fit(X_train, y_train)
        
        loss_curve = numpy.array(best_model.loss_curve_)
        
        n_iterations = len(loss_curve)
        train_acc = numpy.zeros(n_iterations)
        test_acc = numpy.zeros(n_iterations)
        
        # Track model performance over iterations manually - it was seen to be more efficient than using learning_curve
        temp_model = MLPClassifier(random_state=42, warm_start=True, **params)
        temp_model.max_iter = 1
        
        for i in range(n_iterations):
            if i > 0:
                temp_model.max_iter = i+1
            temp_model.fit(X_train, y_train)
            train_acc[i] = temp_model.score(X_train, y_train)
            test_acc[i] = temp_model.score(X_test, y_test)
        
        # # Using learning_curve to get training and testing accuracies
        # _, train_scores, test_scores = learning_curve(
        #     MLPClassifier(random_state=42, **params),
        #     X_train, y_train,
        #     train_sizes=numpy.linspace(0.1, 1.0, n_iterations),
        #     cv=3,
        #     scoring='accuracy',
        #     n_jobs=-1
        # )

        # train_acc = numpy.mean(train_scores, axis=1)
        # test_acc = numpy.mean(test_scores, axis=1)
        
        return best_model, loss_curve, train_acc, test_acc

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """

        # Alphas to work with
        alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        
        standardised_data, _ = self.q2(self.x)
        X_train, X_test, y_train, y_test = train_test_split(standardised_data, self.y, test_size=0.3, random_state=42, stratify=self.y)
        
        train_accuracies = []
        test_accuracies = []
        coefs_norm = []
        train_test_gaps = []
        
        # Creating a base model configuration from optimal_hyperparam
        model_params = {
            'hidden_layer_sizes': optimal_hyperparam['hidden_layer_sizes'][0],
            'activation': optimal_hyperparam['activation'][0],
            'solver' : optimal_hyperparam['solver'][0],
            'learning_rate_init': optimal_hyperparam['learning_rate_init'][0],
            'max_iter': optimal_hyperparam['max_iter'][0]
        }
        
        # Training the model with the different alpha values
        for alpha in alphas:
            model_params['alpha'] = alpha
            model = MLPClassifier(**model_params)
            
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            
            # Storing results
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_test_gaps.append(train_acc - test_acc)
            coef_norm = sum(numpy.linalg.norm(layer) for layer in model.coefs_)
            coefs_norm.append(coef_norm)
            
            print(f"Alpha: {alpha:.6f}, Train: {train_acc:.4f}, Test: {test_acc:.4f}, Norm: {coef_norm:.2f}")
        
        # Return results for visualisation
        return {
            'alphas': alphas,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'coefs_norm': coefs_norm,
            'train_test_gaps': train_test_gaps
        }

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """

        standardised_data, _ = self.q2(self.x)

        params = {}
        for key, value in optimal_hyperparam.items():
            if isinstance(value, list) and len(value) == 1:
                params[key] = value[0]
            else:
                params[key] = value

        mlp = MLPClassifier(**params)

        # Defining the CV methods
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Performing cross-validation
        kf_scores = cross_val_score(mlp, standardised_data, self.y, cv=kf)
        skf_scores = cross_val_score(mlp, standardised_data, self.y, cv=skf)

        kf_mean = numpy.mean(kf_scores)
        skf_mean = numpy.mean(skf_scores)
        
        # Simpler way to calculate p-value
        # _, p_value = stats.ttest_rel(skf_scores, kf_scores)

        n = len(kf_scores)
        difference = skf_scores - kf_scores
        meandiff = numpy.mean(difference)
        stddiff = numpy.std(difference, ddof=1)
        t_statistic = meandiff / (stddiff / numpy.sqrt(n))
        df = n - 1

        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

        if p_value < 0.05:
            result = 'Splitting method impacted performance'
        else:
            result = 'Splitting method had no effect'

        self.cv_results = {
            'kf_scores': kf_scores,
            'skf_scores': skf_scores,
            'result': result
        }
        
        return kf_mean, skf_mean, p_value, result

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """

        standardised_data, _ = self.q2(self.x)

        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=80, random_state=42)
        data_2d = lle.fit_transform(standardised_data)

        unique_labels = numpy.unique(self.y)
        
        results = {
            'embedding': data_2d,
            'labels': self.y,
            'unique_labels': unique_labels,   
            'reconstruction_error': lle.reconstruction_error_
        }
        
        return results