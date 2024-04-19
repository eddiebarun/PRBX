#import statments
from sklearn.metrics import RocCurveDisplay, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Metrics import all_metrics

def main():
    sp500 = sp500_generate()
    sp500 = sp500_time(sp500)
    #import from metrics file with relevant metric
    sp500 = all_metrics(sp500)
    
    print('metrics done')
    
    #removing nan rows from sp500
    sp500.dropna
        
    rand_frst_clf, X_train, X_test, y_train, y_test, y_pred, X_values, y_values, predictors, y_pred_train_data = model(sp500) 
    print('training done')
    
    #maximised output for all columns to be shown 
    #pd.set_option('display.max_columns', None)
    
    print(sp500.tail(30))
    eval_acc_score_training(y_train, y_pred_train_data)
    eval_acc_score(y_test, y_pred)
    eval_class_report(y_test, y_pred)
    eval_conf_matrix(y_test, y_pred)
    fi = eval_feature_importance(rand_frst_clf, X_values)
    eval_fi_graph(rand_frst_clf, fi)
    eval_ROC_curve(rand_frst_clf, X_test, y_test)
    eval_oob_error(rand_frst_clf)
    
    improve_new_rfc(X_train, y_train, X_test, y_test, y_pred, X_values)
    
    print("basic eval done")
    backtest_predictions = eval_backtest(predictors, sp500, X_train, X_test, y_train, y_test, rand_frst_clf)

    print('precision score(back test)',precision_score(backtest_predictions["Target"], backtest_predictions["predictions"]))
    print('accuracy score(back test)', accuracy_score(backtest_predictions["Target"], backtest_predictions["predictions"]))
    print("Finished main")

def sp500_generate():
    sp500_ticker = yf.Ticker("^GSPC")
    sp500 = sp500_ticker.history(period="max")
    sp500["Date"] = sp500.index #Creating a new column that is the date, based off of index which is currently
    sp500["Date"] = pd.to_datetime(sp500["Date"])  # Ensure 'Date' is a datetime format
    
    del sp500["Dividends"] #removing dividends and stock splits as not necessary in index funds
    del sp500["Stock Splits"] 
    
    sp500['Change in Price'] = sp500['Close'].diff() # change in price from today to tomrrow, .diff() subtracst todays close fomr the next rows value.
    
    sp500["Tomorrow"] = sp500["Close"].shift(-1) #Creating a new cloumn that takes the next days close price and puts it as a tomorrow column in todays column 
    
    sp500["Target"] = (sp500["Tomorrow"]>sp500["Close"]).astype(int) #target is just a 0 if tomorrow's close goes down and a 1 if tomorrow close price is up from todays
    
    sp500.dropna
    
    return sp500

def sp500_time(sp500):
    #markets change over time, using data thats too far in the past may cause inaccuracies so we drop data before 1990
    sp500 = sp500.loc["1990-01-01":].copy()
    return sp500

def model(sp500):
    #the predictors used in the model, make sure to update when a new predictor is used. 
    predictors = ['Year','Month','Close','Open', 'High', 'Low','Signal Line','Volume','Change in Price','MA', 'Upper Band', 'Lower Band','RSI', 'K_percent', 'MACD',  'On Balance Volume', 'Price Rate Change', 'Will Percent'] 
    excluded = []
    X_values = sp500[predictors]
    y_values = sp500["Target"]
    
    #improper way of doing training/test splits. 
    #X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, random_state = 0)
    
    #proper way of doing training/test split. 
    #Split X and Y
    rows = len(sp500)
    percent_test = 0.05
     
    X_train= X_values.iloc[:-(round(rows*percent_test))]
    X_test= X_values.iloc[-(round(rows*percent_test)):]
    
    y_train = y_values.iloc[:-(round(rows*percent_test))]
    y_test = y_values.iloc[-(round(rows*percent_test)):]
        
    # Create a Random Forest Classifier  write baout why gini vs other, in video.
    rand_frst_clf = RandomForestClassifier(n_estimators = 200, min_samples_split=10, min_samples_leaf=7, max_depth=30,max_features=None, bootstrap=True, oob_score = True, criterion = "gini", random_state=0)

    # Fit the data to the model
    rand_frst_clf.fit(X_train, y_train)
    
    #original predict function.
    y_pred = rand_frst_clf.predict(X_test)   
    
    y_pred_train_data = rand_frst_clf.predict(X_train)
    
    #Alternate PREDICT FUNCTION (not used
    # Make predictions on all rows
    #y_pred = rand_frst_clf.predict_proba(X_test)[:,1]
    # #custom threshold, instead of buy if chnace greater than 50% changing it to 60% so that model will predict up only if its more confident in preice increase.
    # y_pred[y_pred >= .6] =1
    # y_pred[y_pred < .6] = 0 
    
    # y_pred_train_data = rand_frst_clf.predict_proba(X_train)[:,1]
    # #custom threshold, instead of buy if chnace greater than 50% changing it to 60% so that model will predict up only if its more confident in preice increase.
    # y_pred_train_data[y_pred_train_data < .56] = 0 
    # y_pred_train_data[y_pred_train_data >= .56] = 1   
    
    return rand_frst_clf, X_train, X_test, y_train, y_test, y_pred, X_values, y_values, predictors, y_pred_train_data

def eval_acc_score_training(y_train, y_pred_train_data):
    print('Correct Prediction(for training data) (%): ', accuracy_score(y_train, y_pred_train_data, normalize = True) * 100.0)
    y_pred_train_data = pd.Series(y_pred_train_data, index=y_train.index, name="Predictions")
    print(y_pred_train_data.value_counts())  

def eval_acc_score(y_test, y_pred):
    print('Correct Prediction(for test data) (%): ', accuracy_score(y_test, y_pred, normalize = True) * 100.0)
    y_pred = pd.Series(y_pred, index=y_test.index, name="Predictions")
    print(y_pred.value_counts())
    
def eval_class_report(y_test, y_pred):
    # Define the traget names
    target_names = ['Down Day', 'Up Day']

    # Build a classifcation report
    report = classification_report(y_true = y_test, y_pred = y_pred,target_names=target_names, output_dict = True)

    # Add it to a data frame, transpose it for readability.
    report_df = pd.DataFrame(report).transpose()
  
def eval_conf_matrix(y_test, y_pred):

    "Confusion matrix"
    rf_matrix = confusion_matrix(y_test, y_pred)

    #NB in the confusion matrix look at axis and labels this will tell you where it went right/wrong.
    true_negatives = rf_matrix[0][0]
    false_negatives = rf_matrix[1][0]
    true_positives = rf_matrix[1][1]
    false_positives = rf_matrix[0][1]

    accuracy = (true_negatives + true_positives) / (true_negatives + true_positives + false_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    print('Accuracy: {}'.format(float(accuracy)))
    print('Percision: {}'.format(float(precision)))
    print('Recall: {}'.format(float(recall)))
    print('Specificity: {}'.format(float(specificity)))

    # Create a ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, normalize='true', display_labels = ['Down Day', 'Up Day'], cmap=plt.cm.Blues)
    
    # Customize the plot using various methods provided by ConfusionMatrixDisplay
    # (refer to scikit-learn documentation for details: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)

    # Plot the confusion matrix
    plt.show()
  
def eval_feature_importance(rand_frst_clf, x_values):
    feature_imp = pd.Series(rand_frst_clf.feature_importances_, index=x_values.columns).sort_values(ascending=False)
    print(feature_imp)
    
    return feature_imp

def eval_fi_graph(rand_frst_clf, f_i):
    feature_imp = f_i
    x_values = list(range(len(rand_frst_clf.feature_importances_)))

    # Cumulative importances
    cumulative_importances = np.cumsum(feature_imp.values)

    # Make a line graph
    plt.plot(x_values, cumulative_importances, 'g-')

    # Draw line at 95% of importance retained
    plt.hlines(y = 0.95, xmin = 0, xmax = len(feature_imp), color = 'r', linestyles = 'dashed')

    # Format x ticks and labels
    plt.xticks(x_values, feature_imp.index, rotation = 'vertical')

    # Axis labels and title
    plt.xlabel('Variable')
    plt.ylabel('Cumulative Importance')
    plt.title('Random Forest: Feature Importance Graph') 
    plt.show()

def eval_ROC_curve(rand_frst_clf, X_test, y_test):
    rfc_disp = RocCurveDisplay.from_estimator(rand_frst_clf, X_test, y_test, alpha = 0.8)
    plt.show()
    
def eval_oob_error(rand_frst_clf):
    print('Random Forest Out-Of-Bag Error Score: {}'.format(rand_frst_clf.oob_score_))
    
def eval_pred_vs_real_plot(y_test, y_pred):
    y_pred = pd.Series(y_pred, index = y_test.index)
    pred_real = pd.concat([y_test, y_pred], axis=1)
    pred_real = pred_real.head(50)
    pred_real.plot()
    plt.show()
    
def predict_backtest(X_train, X_test, y_train, y_test, rand_frst_clf):
    rand_frst_clf.fit(X_train, y_train)
    y_pred = rand_frst_clf.predict(X_test)
    y_pred = pd.Series(y_pred, index = y_test.index, name="predictions")
    pred_real = pd.concat([y_test, y_pred], axis=1)
    return pred_real
        
def eval_backtest(predictors, sp500, X_train, X_test, y_train, y_test, rand_frst_clf, start=2500, step=250):
    #building a backtesting function
    #start = the amount of data used to build first model about 250 trading days per year so uses 2500 days or 10 years
    #step = training a model for a year of data then moving to the next year and next year etc. 
    #overall take values for first 10 years predict values for the 11th year then take values for the 11th year and predict values for 12th year and so on....
  
    all_predictions = [] # list of df, each df is the predictions for a single year.
    
    for i in range(start, sp500.shape[0], step):
        train = sp500.iloc[0:i].copy() # training set = all years prior to current year
        test = sp500.iloc[i:(i+step)].copy() # test set = current year.
        
        X_train = train[predictors] #putting training/testing into the format of other functions. 
        y_train = train["Target"]
        X_test = test[predictors] 
        y_test = test["Target"]
        
        if len(X_train) > 0 and len(y_train) > 0: # checking against values errors (Xtrain/ytrain being empty)
            predictions = predict_backtest(X_train, X_test, y_train, y_test, rand_frst_clf)
            all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

def improve_new_rfc(X_train, y_train, X_test, y_test, y_pred, X_values):
    # Number of trees in random forest
    # Number of trees is not a parameter that should be tuned, but just set large enough usually. There is no risk of overfitting in random forest with growing number of # trees, as they are trained independently from each other. 
    n_estimators = list(range(200, 2000, 200))

    # Number of features to consider at every split
    max_features = ['sqrt', None, 'log2']

    # Maximum number of levels in tree
    # Max depth is a parameter that most of the times should be set as high as possible, but possibly better performance can be achieved by setting it lower.
    max_depth = list(range(10, 110, 10))
    max_depth.append(None)

    # Minimum number of samples required to split a node
    # Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree. Too high values can also lead to # under-fitting hence depending on the level of underfitting or overfitting, you can tune the values for min_samples_split.
    min_samples_split = [2, 5, 10, 20, 30, 40]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 7, 12, 14, 16,20]

    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # New Random Forest Classifier to house optimal parameters
    rf = RandomForestClassifier()

    # Specfiy the details of our Randomized Search
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)

    # Fit the random search model
    rf_random.fit(X_train, y_train)
    
    y_pred = rf_random.predict(X_test)


    '''
        ACCURACY
    '''
    # Once the predictions have been made, then grab the accuracy score.
    print('Correct Prediction (%): ', accuracy_score(y_test, rf_random.predict(X_test), normalize = True) * 100.0)


    '''
        CLASSIFICATION REPORT
    '''
    # Define the traget names
    target_names = ['Down Day', 'Up Day']

    # Build a classifcation report
    report = classification_report(y_true = y_test, y_pred = y_pred, target_names = target_names, output_dict = True)

    # Add it to a data frame, transpose it for readability.
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    print('\n')

    '''
        FEATURE IMPORTANCE
    '''
    # Calculate feature importance and store in pandas series
    feature_imp = pd.Series(rf_random.best_estimator_.feature_importances_, index=X_values.columns).sort_values(ascending=False)
    print(feature_imp)
    
    fig, ax = plt.subplots()

    # Create an ROC Curve plot.
    rfc_disp = RocCurveDisplay.from_estimator(rf_random, X_test, y_test, alpha = 0.8, name='ROC Curve', lw=1, ax=ax)

    # Add our Chance Line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Make it look pretty.
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve Random Forest")

    # Add the legend to the plot
    ax.legend(loc="lower right")

    plt.show()
    
    print('best value below')
    print(rf_random.best_estimator_)
    
    
    
        


main()
