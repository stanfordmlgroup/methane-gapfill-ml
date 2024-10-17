from fluxgapfill.main import *
def hello_gapfill():
    # specify the data directory that contains your site
    data_dir = r'./data'
    # specify the directory of the site data
    # we downloaded from: https://ameriflux.lbl.gov/sites/site-search/#vars=FCH4
    # by doing this we actually specify the site directory "./data/CABou"
    sites = 'CABou'
    # specify the model(s) used. Here we use the random forest model as 'rf'.
    # Options of models: ['rf', 'ann', 'lasso', 'xgb'] for
    # random forest, artificial neural network, LASSO, and XGBoost
    models = ['rf']
    #specify the predictors used for gap-filling.
    #Here we used the following four variables as an example
    predictors = ['TA','PA','SW_IN','WS']
    #specify the data source. For example "AmeriFlux-Base"
    #if you process your own data, you can specify data_source as "My_Own"
    data_source = 'AmeriFlux-Base'
    #The function of preprocess is used to seperate the raw.csv into
    # train, validation, and test data for model development.
    # The parameter of "split_method" specify how you split the data.
    # Here we use 'random' method which randomly splits the data
    # For the 'random' method, 'n_mc' parameter is not used and
    # 'n_train' represents the number of times the data is randomly split.
    # For ten-fold cross-validation,the n_train=10
    preprocess(sites=sites,
               na_values=-9999,
               data_source=data_source,
               split_method='random',n_mc=1,n_train=2,
               data_dir=data_dir)
    # Train the machine learning model(s)
    train(sites=sites,
          data_dir=data_dir,
          models=models,
          predictors=predictors,
          overwrite_existing_models=True)
    # Test the machine learning model(s) performance on testing dataset
    # and quantify its uncertainty
    test(data_dir=data_dir,
         sites=sites,
         models=models,
         predictors=predictors,
         split='test',
         distribution='laplace',
         overwrite_results=True
         )
    # Gapfill the CH4 flux using the well-trained machine learning model(s)
    # and quantify its uncertainty. The gap-filled data is saved in "./data/CABou/gapfilled"
    gapfill(data_dir=data_dir,
            sites=sites,
            data_source=data_source,
            predictors=predictors,
            models=models)

if __name__ == "__main__":
    hello_gapfill()
