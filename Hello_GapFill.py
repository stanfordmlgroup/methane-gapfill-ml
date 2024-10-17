from fluxgapfill.main import *
def hello_gapfill():
    data_dir = r'./data'
    sites = 'CABou'#the site data we downloaded from: https://ameriflux.lbl.gov/sites/site-search/#vars=FCH4
    models = ['rf']
    predictors = ['TA','PA','SW_IN','WS']
    data_source = 'AmeriFlux-Base'
    preprocess(sites=sites,
               na_values=-9999,
               data_source=data_source,
               split_method='random',n_mc=1,n_train=2,
               data_dir=data_dir)
    train(sites=sites,
          data_dir=data_dir,
          models=models,
          predictors=predictors,
          overwrite_existing_models=True)
    test(data_dir=data_dir,
         sites=sites,
         models=models,
         predictors=predictors,
         split='test',
         distribution='laplace',
         overwrite_results=True
         )
    gapfill(data_dir=data_dir,
            sites=sites,
            data_source=data_source,
            predictors=predictors,
            models=models)
    run_all(data_dir=data_dir,
            sites=sites,
            models=models,
            predictors='all',
            split='test',
            distribution='laplace',
            overwrite_existing_models=True,
            overwrite_results=True)
if __name__ == "__main__":
    hello_gapfill()
