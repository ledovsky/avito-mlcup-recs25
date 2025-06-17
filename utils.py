import polars as pl
from datetime import timedelta
import time

class TimeTracker:
    def __init__(self, wandb_run=None):
        self.timings = {}
        self.start_times = {}
        self.wandb_run = wandb_run
    
    def start(self, step_name):
        """Start timing a step"""
        self.start_times[step_name] = time.time()
        
    def stop(self, step_name):
        """Stop timing a step and record the elapsed time"""
        if step_name in self.start_times:
            elapsed = time.time() - self.start_times[step_name]
            self.timings[step_name] = elapsed
            print(f"{step_name}: {elapsed:.2f}s")
            
            # Log to wandb if available
            if self.wandb_run:
                self.wandb_run.summary[f"time_{step_name}"] = elapsed
            
            return elapsed
        return None
    
    def print_total(self):
        """Print only the total time of all recorded steps"""
        total = sum(self.timings.values())
        print(f"Total time: {total:.2f}s")


def get_data(data_dir="./data"):

    df_test_users = pl.read_parquet(f'{data_dir}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{data_dir}/clickstream.pq')
    min_date = df_clickstream['event_date'].min()
    df_clickstream = df_clickstream.with_columns(
        ((pl.col('event_date').cast(pl.Date) - pl.lit(min_date)).dt.total_days() // 7 + 1).alias('week')
    )

    df_cat_features = pl.read_parquet(f'{data_dir}/cat_features.pq')
    df_text_features = pl.read_parquet(f'{data_dir}/text_features.pq')
    df_event = pl.read_parquet(f'{data_dir}/events.pq')

    df_clickstream = df_clickstream.join(df_event, on="event")

    EVAL_DAYS_TRESHOLD = 14
    
    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)
    df_train = df_clickstream.filter(df_clickstream['event_date']<= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date']> treshhold)[['cookie', 'node', 'event']]
    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')

    df_eval = df_eval.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact')==1)['event'].unique()
        )
    )

    df_eval = df_eval.filter(
        pl.col('cookie').is_in(df_train['cookie'].unique())
    ).filter(
        pl.col('node').is_in(df_train['node'].unique())
    )

    df_eval = df_eval.unique(['cookie', 'node'])

    return df_test_users, df_clickstream, df_cat_features, df_text_features, df_event, df_train, df_eval



def recall_at(df_true, df_pred, k=40):
    return  df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']], 
        how='left',
        on = ['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum()/pl.col(
                'value'
            ).count()
        ]
    )['value'].mean()


def make_pred_df(df_train:pl.DataFrame, df_eval: pl.DataFrame, preds: pl.Series):
    """Возвращает датафрейм с предсказаниями с дополнительными разрезами"""

    # Оценка бакетов по популярности
    pop_counts = df_train.select("cookie", "node").unique().group_by('node').len().rename({'len':'freq'})
    pop_counts = pop_counts.sort('freq', descending=True)
    total_items = pop_counts.shape[0]
    quantiles = [0.10, 0.30]
    freqs = pop_counts['freq'].to_list()
    th1 = freqs[int(total_items * quantiles[0])]
    th2 = freqs[int(total_items * quantiles[1])]

    # Evaluation by bucket function
    def bucket(freq):
        if freq >= th1:
            return 'high'
        elif freq >= th2:
            return 'medium'
        else:
            return 'low'

    df_true = df_eval.select(['cookie','node']).with_columns(pl.lit(1).alias('true'))
    df_pred = preds.with_columns(pl.lit(1).alias('pred'))
    df_merge = df_true.join(df_pred, on=['cookie','node'], how='left').with_columns(pl.col("pred").fill_nan(0))
    df_merge = df_merge.join(pop_counts, "node", how="left")
    df_merge = df_merge.with_columns(pl.col("freq").map_elements(bucket).alias("bucket"))
    return df_merge
