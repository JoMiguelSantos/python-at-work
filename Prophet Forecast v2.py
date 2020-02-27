"""Customer Service contacts forecast - daily."""
import numpy as np
import pandas as pd
from datetime import timedelta, datetime as dt
from fbprophet import Prophet
import itertools
from utils import df_from_sql,S3Access
from random import sample
import os


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - np.abs(y_pred)) / np.abs(y_pred))) * 100


# perform randomized search for hyperparameters optimization on all forecasts
def random_search(forecasts, languages, contacts):
    random_search_results = {
        "best_params": {
            forecast: {language: None for language in languages}
            for forecast in forecasts
        },
        "best_scores": {
            forecast: {language: None for language in languages}
            for forecast in forecasts
        },
    }

    # set the train test split date and hyperparameters combinations to test
    split_date = (dt.utcnow().date() - timedelta(days=60)).strftime("%d-%b-%Y")
    # Defines the type of growth
    modes = ["additive", "multiplicative"]
    # # Defines the volatility of forecast between points (in this case between days)
    changepoints = [0.001, 0.01, 0.1, 0.5, 1, 2, 5]
    # Define the number of yearly seasonalities (if False then no yearly seasonality)
    yearly_seasonality = [False, 1, 2, 5, 10, 15, 20, 25]
    # Define the monthly seasonality volatility
    fouriers = [15, 12, 10, 5, 3, 1]

    # For each timeframe:
    for forecast in forecasts:
        # get list with different combinations possible to test
        a = [modes, changepoints, yearly_seasonality, fouriers]
        # take 300 random samples to test combinations
        combinations = sample(list(itertools.product(*a)), 300)
        # For each combination
        for combination in combinations:
            # For each language
            for language in languages:

                df_train = filter_df(split_date, language, forecast, contacts)
                df_test = filter_df(split_date, language, forecast, contacts, test=True)

                # initiate model with all parameters from combinations with weekly/monthly/yearly seasonality ON
                model = Prophet(
                    seasonality_mode=combination[0],
                    changepoint_prior_scale=combination[1],
                    yearly_seasonality=combination[2],
                    weekly_seasonality=True,
                )
                # add monthly seasonality
                model.add_seasonality(
                    name="monthly", period=30.5, fourier_order=combination[3]
                )

                # input holidays from particular country if available or TAR (common european holidays) if not available
                if language == "fr":
                    model.add_country_holidays(country_name="FRA")
                    holiday = "FRA"
                elif language != "en":
                    model.add_country_holidays(country_name=language.upper())
                    holiday = language.upper()
                else:
                    model.add_country_holidays(country_name="TAR")
                    holiday = "TAR"

                # fit model with train data
                model.fit(
                    df_train.rename(columns={"period": "ds", "real_contacts": "y"})
                )

                # make a prediction DF using the test data
                df_random = model.predict(
                    df=df_test.reset_index().rename(columns={"period": "ds"})
                )

                # get the mean absolute percentage error for this forecast
                MAPE = mean_absolute_percentage_error(
                    y_true=df_test["real_contacts"], y_pred=df_random["yhat"]
                )
                # exclude runs that give negative predictions due to parameter and data availability interactions
                if len(df_random[df_random["yhat"] < 0]) == 0:
                    # compare the results and update random search results parameters with best MAPE
                    if random_search_results["best_scores"][forecast][language] == None:
                        random_search_results["best_scores"][forecast][language] = MAPE
                        random_search_results["best_params"][forecast][language] = {
                            "seasonality_mode": combination[0],
                            "changepoint_prior_scale": combination[1],
                            "yearly_seasonality": combination[2],
                            "holiday": holiday,
                            "fourier": combination[3],
                        }
                    elif (
                        random_search_results["best_scores"][forecast][language] > MAPE
                    ):
                        random_search_results["best_params"][forecast][language] = {
                            "seasonality_mode": combination[0],
                            "changepoint_prior_scale": combination[1],
                            "yearly_seasonality": combination[2],
                            "holiday": holiday,
                            "fourier": combination[3],
                        }
                        random_search_results["best_scores"][forecast][language] = MAPE
                else:
                    continue
    return random_search_results


# pipeline to get forecast production data using the parameters set by the random search
def get_forecast(random_search_results, forecasts, languages, contacts):
    split_date = dt.utcnow().date().strftime("%d-%b-%Y")
    df_fcst = {
        forecast: {language: None for language in languages} for forecast in forecasts
    }
    for forecast in forecasts:
        for language in languages:

            # get all parameters loaded from the random search results for each language
            fourier = random_search_results["best_params"][forecast][language][
                "fourier"
            ]
            yearly = random_search_results["best_params"][forecast][language][
                "yearly_seasonality"
            ]
            changepoint = random_search_results["best_params"][forecast][language][
                "changepoint_prior_scale"
            ]
            mode = random_search_results["best_params"][forecast][language][
                "seasonality_mode"
            ]

            df_train = filter_df(split_date, language, forecast, contacts)

            # initiate model with all parameters from random search results
            model = Prophet(
                seasonality_mode=random_search_results["best_params"][forecast][
                    language
                ]["seasonality_mode"],
                changepoint_prior_scale=random_search_results["best_params"][forecast][
                    language
                ]["changepoint_prior_scale"],
                yearly_seasonality=random_search_results["best_params"][forecast][
                    language
                ]["yearly_seasonality"],
                weekly_seasonality=True,
            )
            model.add_seasonality(
                name="monthly",
                period=30.5,
                fourier_order=random_search_results["best_params"][forecast][language][
                    "fourier"
                ],
            )

            # input holidays from particular country if available or TAR (common european holidays) if not available
            if language == "fr":
                model.add_country_holidays(country_name="FRA")
                holiday = "FRA"
            elif language != "en":
                model.add_country_holidays(country_name=language.upper())
                holiday = language.upper()
            else:
                model.add_country_holidays(country_name="TAR")
                holiday = "TAR"

            # fit model with train data
            model.fit(df_train.rename(columns={"period": "ds", "real_contacts": "y"}))

            # create DF with future dates to forecast
            future = model.make_future_dataframe(
                freq="D", periods=365, include_history=False
            )

            # separate weekdays from saturday live forecast
            if "live" in forecast:
                if "saturday" in forecast:
                    future = future[future["ds"].dt.dayofweek.isin([5])]
                else:
                    future = future[~future["ds"].dt.dayofweek.isin([6, 5])]
            # make prediction DF
            df_fcst[forecast][language] = model.predict(future)

            # add channel, language and model date columns
            df_fcst[forecast][language]["language"] = language
            if "live" in forecast:
                df_fcst[forecast][language]["channel"] = "live"
            else:
                df_fcst[forecast][language]["channel"] = "email"
    return df_fcst


def filter_df(split_date, language, forecast, contacts, test=False):
    # apply different trasnsformations depending on channel
    if "live" in forecast:
        channel = ["chat", "call"]
        # filter for language and channels and group call and chat into live
        # If UK then only include days after official public launch
        if language == "uk":
            df = (
                contacts[
                    (contacts["channel"].isin(channel))
                    & (contacts["contact_language"] == language)
                    & (
                        contacts["period"] > dt.strptime("2018-11-14", "%Y-%m-%d")
                    )  # UK started the public launch by Nov 14, 2018
                ]
                .groupby(["period", "contact_language"])
                .sum()
                .reset_index()
            )
        else:
            df = (
                contacts[
                    (contacts["channel"].isin(channel))
                    & (contacts["contact_language"] == language)
                ]
                .groupby(["period", "contact_language"])
                .sum()
                .reset_index()
            )

        # split saturdays and weekdays on live forecast
        if "saturday" in forecast:
            df_train = df[
                (df["period"] < split_date) & (df["period"].dt.dayofweek.isin([5]))
            ][["period", "real_contacts"]]
            df_test = df[
                (df["period"] > split_date) & (df["period"].dt.dayofweek.isin([5]))
            ][["period", "real_contacts"]]
        else:
            df_train = df[
                (df["period"] < split_date) & (~df["period"].dt.dayofweek.isin([6, 5]))
            ][["period", "real_contacts"]]
            df_test = df[
                (df["period"] > split_date) & (~df["period"].dt.dayofweek.isin([6, 5]))
            ][["period", "real_contacts"]]

    elif "email" in forecast:
        channel = ["email"]
        if language == "uk":
            df = (
                contacts[
                    (contacts["channel"].isin(channel))
                    & (contacts["contact_language"] == language)
                    & (contacts["period"] > dt.strptime("2018-11-14", "%Y-%m-%d"))
                ]
                .groupby(["period", "contact_language"])
                .sum()
                .reset_index()
            )
        else:
            df = (
                contacts[
                    (contacts["channel"].isin(channel))
                    & (contacts["contact_language"] == language)
                ]
                .groupby(["period", "contact_language"])
                .sum()
                .reset_index()
            )
        df_train = df[(df["period"] < split_date)][["period", "real_contacts"]]
        df_test = df[(df["period"] > split_date)][["period", "real_contacts"]]
    if test:
        return df_test
    return df_train


def send_to_DWH(df_fcst, languages, forecasts):
    for language in languages:
        df_fcst = pd.concat(
            [
                df_fcst[forecast][language][
                    ["ds", "language", "channel", "yhat", "yhat_upper", "yhat_lower"]
                ]
                for forecast in forecasts
            ],
            axis=0,
        )
        # Save to S3
        now = dt.utcnow()
        now_f = now.strftime("%Y_%m_%d_%H_%M_%s")
        path = os.path.join("custom_path)

        filename = f"{language}{now_f}.tsv"

        df_fcst["model_date"] = now.strftime("%Y_%m_%d")
        df_fcst = df_fcst[df_fcst["ds"] > now]
        tsv = df_fcst.to_csv(
            filename,
            sep="\t",
            columns=[
                "model_date",
                "language",
                "channel",
                "ds",
                "yhat",
                "yhat_upper",
                "yhat_lower",
            ],
            header=False,
            index=False,
        )
        S3Access.send_to_s3("etl_path_example", path, "", filename, zip=False)


def generate_forecasts():
    # setup evaluation variables and initial values
    languages = ["de", "fr", "en", "es", "it"]
    forecasts = ["daily_live", "daily_live_saturday", "daily_email"]
    # get df with contacts
    contacts = df_from_sql("contacts.sql")
    contacts["period"] = pd.to_datetime(contacts["period"])
    # perform random search and obtain best parameters
    random_search_results = random_search(forecasts, languages, contacts)
    # get final forecast using best parameters
    df_fcst = get_forecast(random_search_results, forecasts, languages, contacts)
    # send forecast to DWH
    send_to_DWH(df_fcst, languages, forecasts)
    return df_fcst


if __name__ == "__main__":
    generate_forecasts()