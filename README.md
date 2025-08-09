# Data-Mining
<p>The following repository holds a data mining project where an end-to-end image classification pipeline was implemented using a Multi-Layer Perceptron (MLP) with hold-out validation, including manual hyperparameter tuning and performance visualization.</p>

<h2>Dataset Loading</h2>
<p>The first step was to load the image classification dataset into the environment as per the specification in the q1 function of `coc131_cw.py`. Since the images were already stored locally in a structured format, I implemented file-handling logic to iterate through directories, read each image, and assign the correct class label based on its folder name. The Pillow library was used for image file loading and initial processing. Each image was converted into a consistent format (e.g., RGB) and reshaped into a flat array suitable for input into the machine learning model. This ensured uniformity across samples while preserving key pixel-level features.</p>

<h2>Data Standardization and Visualization</h2>
<p>First part: To standardize the dataset, I implemented a function in q2 that adjusted each feature to have zero mean and unit variance. The formula applied was:</p>
<p><strong>x′ = (x−μ) / σ</strong></p>
<p>Where `𝜇` is the feature mean and `σ` is the feature standard deviation. This was performed across all features to ensure that no single dimension disproportionately influenced the model during training.</p>
<img width="500" height="600" alt="q2-1" src="https://github.com/user-attachments/assets/3e0afb97-af50-4faf-95e2-712aaff0c77f" />
<p>Second part: To visualize the impact of standardization, I selected representative samples from the dataset and plotted them before and after the transformation using matplotlib. Additionally, feature distribution histograms were plotted to clearly demonstrate the compression of feature variance and the centering of the mean at zero. This confirmed that the data preprocessing step was correctly implemented.</p>
<img width="3568" height="1466" alt="q2-2" src="https://github.com/user-attachments/assets/7cc87937-4e60-447d-86b7-270af8c821ab" />

<h2>Multi-Layer Perceptron (MLP) Classifier</h2>
<p>First Part: Using `sklearn.neural_network.MLPClassifier`, I built a Multi-Layer Perceptron for image classification. The dataset was split using the hold-out method, reserving 30% of samples for testing. The architecture and solver parameters were chosen based on iterative experimentation, adjusting the number of hidden layer neurons, activation functions, and the maximum number of iterations until the model converged.</p>
<img width="4764" height="1899" alt="q3-1" src="https://github.com/user-attachments/assets/f1919641-ea99-4c11-a179-03e3b8d5dd04" />

<p>Second Part: Optimal hyperparameters were determined through a manual search strategy. I systematically varied parameters such as the hidden layer size, learning rate (alpha), and solver type while observing classification accuracy. The final chosen parameters were stored in the optimal_hyperparam variable in coc131_cw.py for reproducibility.</p>
<img width="2140" height="1963" alt="q3-2" src="https://github.com/user-attachments/assets/c1183b44-03da-4938-a118-5a588ee9b920" />

<p>Third Part: Performance across hyperparameter configurations was summarized using accuracy vs. parameter value plots. These visualizations allowed for clear identification of trends and optimal configurations, aiding in the final model selection.</p>

<h2>Impact of Alpha (Learning Rate Regularization Term)</h2>
<p>First Part: The influence of the regularization parameter alpha on both model parameters (weights and biases) and classification performance was systematically tested. By varying alpha over a defined range, I trained separate MLP instances and extracted their learned weights and biases using the model’s attributes (coefs_ and intercepts_).</p>
<img width="5370" height="1752" alt="q4-1" src="https://github.com/user-attachments/assets/023db3aa-40be-48e5-907d-0ff7446aa647" />

<p>Second Part: The results were visualized using line plots for accuracy vs. alpha, alongside weight magnitude plots for each model configuration. These plots clearly highlighted the trade-off between overfitting (low alpha) and underfitting (high alpha), demonstrating how regularization constrains weight values and impacts generalization performance.</p>
<img width="2964" height="1750" alt="q4-2" src="https://github.com/user-attachments/assets/e6f135ee-84e4-4a1a-9c8e-6712bda68bbc" />

<h2>Cross-Validation with and without Stratification</h2>
<p>First Part: I conducted a comparative analysis using 5-fold cross-validation with and without stratified sampling. For stratified CV, class proportions were preserved in each fold, whereas in standard CV, samples were randomly assigned to folds without regard to class balance. Classification accuracy from each fold was recorded for both methods.</p>
<img width="3564" height="2969" alt="q5-1" src="https://github.com/user-attachments/assets/5bce8aa1-2bcb-4652-b796-65f9ab5424eb" />

<p>Second Part: A paired hypothesis test (two-tailed t-test) was performed on the accuracy results to determine whether stratification produced a statistically significant improvement. The resulting p-values and test statistics were reported, confirming or rejecting the null hypothesis. Accuracy boxplots were then generated for both conditions, providing a visual summary of the results and supporting statistical conclusions.</p>
<img width="2958" height="1753" alt="q5-2" src="https://github.com/user-attachments/assets/bd82bdf7-813a-49d0-aa2f-bad31f1afd6c" />

<h2>Dimensionality Reduction with Locally Linear Embedding (LLE)</h2>
<p>For the final task, LocallyLinearEmbedding from sklearn.manifold was applied to the standardized dataset to reduce the high-dimensional image features to a 2D representation. LLE preserves local neighborhood relationships, making it effective for visualizing class separability in datasets with complex manifolds. The resulting 2D embeddings were plotted with each point colored according to its class label. This visualization provided an intuitive representation of how well the classes could be separated in the transformed space, highlighting inherent structure in the dataset beyond what was observable in raw pixel space.</p>
<img width="4170" height="2980" alt="q6-1" src="https://github.com/user-attachments/assets/0a264e11-6367-451f-96b8-9ae8fe63e709" />
<img width="5984" height="2955" alt="q6-2" src="https://github.com/user-attachments/assets/bc0d926c-38d6-43ef-b1b3-0112df08a20d" />
