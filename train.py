import torch
import numpy as np
import pandas as pd
import src.utils as utils
import optuna

DEVICE = "cuda"
EPOCHS = 150

def run_training(folds,params,save_model):
    df = pd.read_csv(r'C:\Users\Jeev\PycharmProjects\Proj1\input\train_features.csv')
    df = df.drop(['cp_type','cp_time','cp_dose'],axis=1)

    targets_df = pd.read_csv(r'C:\Users\Jeev\PycharmProjects\Proj1\input\train_targets_fold.csv')

    feature_columns = df.drop("sig_id",axis=1).columns
    target_columns = targets_df.drop(["sig_id","kfold"], axis=1).columns

    df = df.merge(targets_df,on="sig_id",how="left")

    train_df = df[df.kfold != folds].reset_index(drop=True)
    valid_df = df[df.kfold == folds].reset_index(drop=True)

    x_train = train_df[feature_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()
    x_valid = valid_df[feature_columns].to_numpy()
    y_valid = valid_df[target_columns].to_numpy()

    train_dataset = utils.MOAData(x_train,y_train)
    valid_dataset = utils.MOAData(x_valid,y_valid)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = 1024,shuffle = True,num_workers = 8)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size = 1024,num_workers = 8)

    model = utils.Model(x_train.shape[1],y_train.shape[1],params["num_layers"],params["hidden_size"],params["dropout"])

    model.to(DEVICE)
    optim = torch.optim.Adam(model.parameters(),lr=params["learning_rate"])
    eng = utils.Engine(model=model,optimizer=optim,device=DEVICE)

    early_stopping_iter = 10
    best_loss = np.inf
    early_stopping_counter = 0
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_dataloader)
        valid_loss = eng.eval(valid_dataloader)

        print(f"{folds}. {epoch}, {train_loss}, {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(),f"model_{folds}.bin")

        else:
            early_stopping_counter +=1
        if early_stopping_counter>early_stopping_iter:
            break

    return best_loss
def objective(trial):
    params = {
        "num_layers": trial.suggest_int("num_layers", 1, 7),
        "hidden_size":trial.suggest_int("hidden_size", 16, 2380),
        "dropout":trial.suggest_int("dropout", 0.1, 0.7, 0.3),
        "learning_rate":trial.suggest_int("learning_rate", 1e-6, 1e-4)

        }

    all_losses = []
    for f in range(10):
        temp_loss = run_training(f,params,save_model=True)
        all_losses.append(temp_loss)


if __name__ == '__main__':
    params = {
        "num_layers": 10,
        "hidden_size": 2380,
        "dropout":  0.3,
        "learning_rate": 0.001

    }
    run_training(folds=0,save_model=True,params=params)
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective,n_trials=20)
    #
    # print("best_ :")
    # trial_ = study.best_trial
    #
    # print(trial_.params)
    # print(trial_.value)
    #
    # scores = 0
    # for i in range(10):
    #     scr = run_training(j,trial_.params,save_model=True)
    #     scores += scr
    #
    # print(scores/5)