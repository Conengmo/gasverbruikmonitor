import csv
import datetime as dt
import math
import os
from typing import List, Tuple, Dict, Any, Iterator

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils import shuffle
from tqdm import tqdm

LOWEST_DATE = dt.date(2019, 6, 20)
# INTERVENTION = dt.date(2021, 10, 16)
INTERVENTION = dt.date(2022, 1, 16)
HIGHEST_DATE = dt.date(2022, 9, 5)

REMOVE_DATES = [
    dt.date(2021, 11, 28),
    dt.date(2021, 11, 29),
    dt.date(2021, 11, 30),
    dt.date(2021, 12, 1),
]


def read_gasverbruik() -> Dict[dt.date, int]:
    path = 'data/meterstanden_gas/'
    gas_stand = {}
    for filename in os.listdir(path):
        with open(f'{path}/{filename}') as f:
            for item in csv.DictReader(f, delimiter=';'):
                date = dt.datetime.strptime(item['OpnameDatum'], '%Y-%m-%d').date()
                verbruik = int(item['Gas'])
                gas_stand[date] = verbruik

    dates = list(sorted(gas_stand))
    standen = [stand for _, stand in sorted(gas_stand.items())]
    verbruiken: List[int] = np.diff(standen).tolist()  # type: ignore
    verbruiken.append(verbruiken[-1])
    verbruik_dict = dict(zip(dates, verbruiken))
    return clean_dates(verbruik_dict)


def read_weather():
    weather_data = dict(
        min_temp={},
        max_temp={},
        # sunshine_duration={},
        radiation={},
        # wind_direction={},
        wind_speed={},
        # wind_zero={},
        # wind_north={},
        # wind_east={},
        # wind_south={},
        # wind_west={},
    )
    with open('data/weer/voorschoten.csv') as f:
        for item in csv.DictReader(f):
            date = dt.datetime.strptime(item['YYYYMMDD'], '%Y%m%d').date()
            if date < LOWEST_DATE or date > HIGHEST_DATE:
                continue
            wind_dir = int(item['DDVEC'])
            weather_data['min_temp'][date] = int(item['TN'])
            weather_data['max_temp'][date] = int(item['TX'])
            # weather_data['sunshine_duration'][date] = int(item['SQ'])
            weather_data['radiation'][date] = int(item['Q'])
            # weather_data['wind_direction'][date] = int(item['DDVEC'])
            weather_data['wind_speed'][date] = int(item['FG'])
            # weather_data['wind_zero'][date] = int(wind_dir == 0)
            # weather_data['wind_north'][date] = int(item['FG']) * (
            #     _get_wind_dir_val(wind_dir - 360 if wind_dir > 180 else wind_dir, -90, 90) if wind_dir != 0 else 0
            # )
            # weather_data['wind_east'][date] = int(item['FG']) * _get_wind_dir_val(wind_dir, 0, 180)
            # weather_data['wind_south'][date] = int(item['FG']) * _get_wind_dir_val(wind_dir, 90, 270)
            # weather_data['wind_west'][date] = int(item['FG']) * _get_wind_dir_val(wind_dir, 180, 360)

    return {key: clean_dates(values) for key, values in weather_data.items()}


def _get_wind_dir_val(val, lower, higher) -> int:
    return 1 + abs(val - (lower + 90)) / -90 if lower < val < higher else 0


def iter_dates() -> Iterator[dt.date]:
    date = LOWEST_DATE
    while date <= HIGHEST_DATE:
        yield date
        date += dt.timedelta(days=1)


def create_is_weekend_data():
    out = {date: int(date.isoweekday() in [6, 7]) for date in iter_dates()}
    return out


def create_distance_from_new_year():
    half_year = 365 // 2
    out = {date: abs(abs(date.timetuple().tm_yday - half_year) - half_year) for date in iter_dates()}
    return out


def clean_dates(data: Dict[dt.date, Any]) -> Dict[dt.date, Any]:
    out = {}
    date = LOWEST_DATE
    while date <= HIGHEST_DATE:
        out[date] = data.get(date)
        if out[date] is None:
            out[date] = data[date - dt.timedelta(days=1)]
        date += dt.timedelta(days=1)
    return out


def smooth(y):
    n = len(y)
    # flatten summer
    out = [
        0 if idx < n - 5 and y[idx - 5:idx + 5].sum() <= 2 else val
        for idx, val in enumerate(y)
    ]
    return np.array(out)


import torch
from torch.autograd import Variable


class CustomMixin:

    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def init_scaler(self, samples):
        self.scaler.fit(samples)

    def predict(self, samples):
        with torch.no_grad():
            return np.ravel(self(Variable(torch.from_numpy(self.scaler.transform(samples))).float()).data.numpy())


class LinearRegression(CustomMixin, torch.nn.Module):

    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


class MLP(CustomMixin, torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


def pytorch_train(x_train, y_train, x_test, y_test):
    input_dim = x_train.shape[1]
    output_dim = 1
    learning_rate = 0.001
    epochs = 10000

    # model = LinearRegression(input_dim, output_dim)
    # criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model = MLP(input_dim, output_dim)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.init_scaler(x_train)
    x_train = model.scaler.transform(x_train)
    x_test = model.scaler.transform(x_test)
    y_train = y_train[:, None]
    y_test = y_test[:, None]
    test_inputs = Variable(torch.from_numpy(x_test)).float()
    test_labels = Variable(torch.from_numpy(y_test)).float()

    model_state = model.state_dict()
    lowest_test_loss = math.inf
    epoch_of_saved_state = 0
    pbar = tqdm(epochs)
    for epoch in range(epochs):
        pbar.update()
        # Converting inputs and labels to Variable
        _x_train, _y_train = shuffle(x_train, y_train, random_state=42)
        inputs = Variable(torch.from_numpy(_x_train)).float()
        labels = Variable(torch.from_numpy(_y_train)).float()
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
        # get output from the model, given the inputs
        outputs = model(inputs)
        # get loss for the predicted output
        loss = criterion(outputs, labels)
        # get gradients w.r.t to parameters
        loss.backward()
        # update parameters
        optimizer.step()
        # calculate test loss
        with torch.no_grad():
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels).item()
        if test_loss < lowest_test_loss:
            model_state = model.state_dict()
            lowest_test_loss = test_loss
            epoch_of_saved_state = epoch
        pbar.set_postfix(train_loss=loss.item(), test_loss=test_loss)

    print(f'load state {epoch_of_saved_state} with test loss', lowest_test_loss)
    model.load_state_dict(model_state)

    return model


def main():
    gas_verbruik = read_gasverbruik()
    weather = read_weather()
    is_weekend = create_is_weekend_data()
    distance_from_new_year = create_distance_from_new_year()
    dates = list(iter_dates())

    samples = np.array([
        # list(is_weekend.values()),
        # list(distance_from_new_year.values()),
        *[list(entry.values()) for entry in weather.values()],
    ]).T
    targets = np.array(list(gas_verbruik.values()))
    targets = smooth(targets)

    n_pred = (HIGHEST_DATE - INTERVENTION).days
    _temp_samples, pred_samples, _temp_targets, pred_targets, _temp_dates, pred_dates = \
        train_test_split(samples, targets, dates, test_size=n_pred, shuffle=False)
    train_samples, test_samples, train_targets, test_targets, train_dates, test_dates = \
        train_test_split(_temp_samples, _temp_targets, _temp_dates, train_size=0.8, shuffle=True)

    # Bayesian Regression
    # clf = linear_model.BayesianRidge()
    # clf.fit(train_samples, train_targets)

    # Support Vector Machine
    clf = make_pipeline(StandardScaler(), SVR())
    clf.fit(train_samples, train_targets)

    # Pytorch
    # clf = pytorch_train(train_samples, train_targets, test_samples, test_targets)

    train_result = clf.predict(train_samples)
    test_result = clf.predict(test_samples)
    pred_result = clf.predict(pred_samples)
    all_result = clf.predict(samples)

    train_diff = train_targets - train_result
    test_diff = test_targets - test_result
    pred_diff = pred_targets - pred_result
    all_diff = targets - all_result
    train_diff[np.isin(train_dates, REMOVE_DATES)] = 0
    test_diff[np.isin(test_dates, REMOVE_DATES)] = 0
    pred_diff[np.isin(pred_dates, REMOVE_DATES)] = 0
    all_diff[np.isin(dates, REMOVE_DATES)] = 0
    print(f'train diff: {np.mean(train_diff):.2f}')
    print(f'test diff: {np.mean(test_diff):.2f}')
    print(f'pred diff: {np.mean(pred_diff):.2f}')
    print(f'overall diff: {np.mean(all_diff):.2f}')

    fix, ax = plt.subplots()
    ax.plot(train_dates, train_diff, '.', label='train')
    ax.plot(test_dates, test_diff, '.', label='test')
    ax.plot(pred_dates, pred_diff, '.', label='test')
    ax.hlines(0, xmin=dates[0], xmax=dates[-1], colors='grey', linestyle='--')
    ax.legend()

    kernel = np.ones(7) / 7
    all_diff_convolved = np.convolve(all_diff, kernel, mode='same')
    fix, ax = plt.subplots()
    ax.plot(dates, all_diff_convolved)
    ax.hlines(0, xmin=dates[0], xmax=dates[-1], colors='grey', linestyle='--')

    # fix, ax = plt.subplots()
    # ax.plot(dates, gas_verbruik.values(), '.')
    # ax.plot(dates, targets, '-')

    # plt.plot(distance_from_new_year.keys(), distance_from_new_year.values(),  '.')
    # plt.plot(weather.keys(), [p.wind_direction for p in weather.values()], '.')

    # plt.plot(range(0, 360), [_get_wind_dir_val(val, 90, 270) for val in range(0, 360)], '.')

    # fix, axs = plt.subplots(4, 1, sharex=True)
    # axs[0].plot(range(0, 361), [_get_wind_dir_val(wind_dir - 360 if wind_dir > 180 else wind_dir, -90, 90) \
    #                  if wind_dir != 0 else 0 for wind_dir in range(361)], '.'),
    # axs[1].plot(range(361), [_get_wind_dir_val(wind_dir, 0, 180) for wind_dir in range(361)], '.')
    # axs[2].plot(range(361), [_get_wind_dir_val(wind_dir, 90, 270) for wind_dir in range(361)], '.')
    # axs[3].plot(range(361), [_get_wind_dir_val(wind_dir, 180, 360) for wind_dir in range(361)], '.')

    # fig, axs = plt.subplots(3, 1, sharex=True)
    # axs[0].plot(gas_verbruik.keys(), gas_verbruik.values(), '.')
    # # axs[1].plot(weather.keys(), [p.min_temp for p in weather.values()], '.')
    # # axs[1].plot(weather.keys(), [p.max_temp for p in weather.values()], '.')
    # axs[1].plot(is_weekend.keys(), is_weekend.values(), '.')
    # axs[2].plot(weather['min_temp'].keys(), weather['min_temp'].values(), '.')

    plt.show()


if __name__ == '__main__':
    main()
