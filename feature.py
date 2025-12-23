import pandas as pd
import numpy as np
import warnings
from scipy import stats
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

TRAIN_FILE = "master_train_corrected.parquet"
TEST_FILE = "master_test_corrected.parquet"
OUTPUT_TRAIN = "train_features.csv"
OUTPUT_TEST = "test_features.csv"

def measure_durations(flux, time, peak_idx, min_f, amplitude, pct=0.15):
    threshold = min_f + (pct * amplitude)
    
    pre_peak_flux = flux[:peak_idx]
    pre_peak_time = time[:peak_idx]
    below_indices = np.where(pre_peak_flux <= threshold)[0]
    
    if len(below_indices) > 0:
        start_idx = below_indices[-1]
        rise_dur = time[peak_idx] - pre_peak_time[start_idx]
    else:
        rise_dur = time[peak_idx] - time[0]
        if pct < 0.3 and rise_dur > 200:
            rise_dur = 200 

    post_peak_flux = flux[peak_idx+1:]
    post_peak_time = time[peak_idx+1:]
    below_indices_post = np.where(post_peak_flux <= threshold)[0]
    
    if len(below_indices_post) > 0:
        end_idx = below_indices_post[0]
        fall_dur = post_peak_time[end_idx] - time[peak_idx]
    else:
        fall_dur = time[-1] - time[peak_idx]
        if pct < 0.3 and fall_dur > 300:
            fall_dur = 300 
        
    return rise_dur, fall_dur

def measure_activity(flux, flux_err, peak_idx, region='pre'):
    if region == 'pre':
        if peak_idx < 3:
            return 0, 0
        segment_flux = flux[:peak_idx]
        segment_err = flux_err[:peak_idx] + 1e-9
    else:
        if len(flux) - peak_idx < 3:
            return 0, 0
        segment_flux = flux[peak_idx+1:]
        segment_err = flux_err[peak_idx+1:] + 1e-9
        
    snr = segment_flux / segment_err
    n_bumps = np.sum(snr > 3.0)
    variability = np.mean(snr**2)
    return n_bumps, variability

def get_time_series_features(x):
    x = x.sort_values('Time (MJD)')
    flux = x['Flux_Corrected'].values
    if 'Flux_err_Corrected' in x.columns:
        flux_err = x['Flux_err_Corrected'].values
    else:
        flux_err = x['Flux_err'].values
    time = x['Time (MJD)'].values
    
    feat_names = [
        'mean', 'max', 'min', 'std', 't_peak',
        'rise_15', 'fall_15', 'rise_50', 'fall_50',
        'amplitude', 'neumann', 'rise_slope_15', 'fwhm', 'plateau_slope',
        'pre_n_bumps', 'pre_variability',
        'post_n_bumps', 'post_variability',
        'tail_neumann', 'decay_slope', 'last_flux',
        'skewness', 'kurtosis', 'median', 'iqr', 'mad',
        'peak_significance', 'n_peaks', 'cadence_std',
        'rise_acceleration', 'fall_acceleration'
    ]

    if len(flux) < 3:
        return pd.Series(0, index=feat_names)

    mean_f = np.mean(flux)
    max_f = np.max(flux)
    min_f = np.min(flux)
    std_f = np.std(flux)
    amplitude = (max_f - min_f) / 2
    
    peak_idx = np.argmax(flux)
    t_peak = time[peak_idx]
    
    if len(flux) >= 3:
        last_flux = np.mean(flux[-3:])
    else:
        last_flux = flux[-1]

    r15, f15 = measure_durations(flux, time, peak_idx, min_f, (max_f - min_f), pct=0.15)
    r50, f50 = measure_durations(flux, time, peak_idx, min_f, (max_f - min_f), pct=0.50)
    
    if std_f > 0:
        neumann = np.mean(np.diff(flux)**2) / (np.var(flux) + 1e-9)
    else:
        neumann = 2.0

    if r15 > 1:
        rise_slope_15 = amplitude / r15
    else:
        rise_slope_15 = 0

    fwhm = r50 + f50
    
    if f50 > 1:
        plateau_slope = (amplitude * 0.5) / f50
    else:
        plateau_slope = 0
    
    decay_slope = plateau_slope

    pre_n_bumps, pre_variability = measure_activity(flux, flux_err, peak_idx, 'pre')
    post_n_bumps, post_variability = measure_activity(flux, flux_err, peak_idx, 'post')
    
    post_peak = flux[peak_idx:]
    if len(post_peak) > 5:
        tail_neumann = np.mean(np.diff(post_peak)**2) / (np.var(post_peak) + 1e-9)
    else:
        tail_neumann = 2.0
    
    skewness = stats.skew(flux) if len(flux) > 3 else 0
    kurtosis = stats.kurtosis(flux) if len(flux) > 3 else 0
    median_f = np.median(flux)
    iqr = np.percentile(flux, 75) - np.percentile(flux, 25)
    mad = np.median(np.abs(flux - median_f))
    
    snr = flux / (flux_err + 1e-9)
    peak_significance = snr[peak_idx] if len(snr) > 0 else 0
    
    if len(flux) > 5:
        peaks, _ = find_peaks(flux, height=np.percentile(flux, 70))
        n_peaks = len(peaks)
    else:
        n_peaks = 1
    
    if len(time) > 2:
        time_diffs = np.diff(time)
        cadence_std = np.std(time_diffs)
    else:
        cadence_std = 0
    
    if peak_idx > 2:
        rise_flux = flux[:peak_idx]
        rise_time = time[:peak_idx]
        if len(rise_flux) > 2:
            rise_grad = np.gradient(rise_flux, rise_time)
            rise_acceleration = np.mean(np.diff(rise_grad))
        else:
            rise_acceleration = 0
    else:
        rise_acceleration = 0
    
    if len(flux) - peak_idx > 3:
        fall_flux = flux[peak_idx:]
        fall_time = time[peak_idx:]
        if len(fall_flux) > 2:
            fall_grad = np.gradient(fall_flux, fall_time)
            fall_acceleration = np.mean(np.diff(fall_grad))
        else:
            fall_acceleration = 0
    else:
        fall_acceleration = 0

    return pd.Series([
        mean_f, max_f, min_f, std_f, t_peak,
        r15, f15, r50, f50,
        amplitude, neumann, rise_slope_15, fwhm, plateau_slope,
        pre_n_bumps, pre_variability,
        post_n_bumps, post_variability,
        tail_neumann, decay_slope, last_flux,
        skewness, kurtosis, median_f, iqr, mad,
        peak_significance, n_peaks, cadence_std,
        rise_acceleration, fall_acceleration
    ], index=feat_names)

def calculate_coincidence(df_raw):
    if 'Flux_err_Corrected' in df_raw.columns:
        snr = df_raw['Flux_Corrected'] / (df_raw['Flux_err_Corrected'] + 1e-9)
    else:
        snr = df_raw['Flux'] / (df_raw['Flux_err'] + 1e-9)

    dets = df_raw[snr > 5.0].copy()
    dets['date'] = pd.to_datetime(dets['Time (MJD)'], unit='D', origin=pd.Timestamp('1858-11-17'))
    filter_map = {'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'y': 6}
    dets['filter_id'] = dets['Filter'].map(filter_map)
    dets = dets[['object_id', 'date', 'filter_id']].sort_values(['object_id', 'date'])
    
    if dets.empty:
        return pd.Series(0, name='detection_coincidence')

    def count_unique_bands(sub):
        return sub.rolling('10D', on='date')['filter_id'].apply(lambda x: len(set(x)))
    
    coincidence = dets.groupby('object_id').apply(count_unique_bands)
    max_coincidence = coincidence.groupby(level=0).max()
    max_coincidence.name = 'detection_coincidence'
    return max_coincidence

def calculate_peak_overlap(row):
    events = []
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    for b in bands:
        if f"{b}_max" not in row or row[f"{b}_max"] <= 0:
            continue
        t_peak = row.get(f"{b}_t_peak", 0)
        rise = row.get(f"{b}_rise_15", 0)
        fall = row.get(f"{b}_fall_15", 0)
        if t_peak == 0:
            continue
        events.append((t_peak - rise, 1))
        events.append((t_peak + fall, -1))
    if not events:
        return 0
    events.sort(key=lambda x: x[0])
    max_overlap = 0
    current_overlap = 0
    for _, change in events:
        current_overlap += change
        max_overlap = max(max_overlap, current_overlap)
    return max_overlap

def make_features(input_path, output_path, is_train=True):
    try:
        df = pd.read_parquet(input_path)
    except:
        return

    grouped = df.groupby(['object_id', 'Filter']).apply(get_time_series_features).unstack()
    grouped.columns = [f"{col[1]}_{col[0]}" for col in grouped.columns]
    
    meta_cols = ['object_id', 'Z', 'Z_err', 'EBV', 'target', 'truth', 'SpecType']
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].drop_duplicates('object_id').set_index('object_id')
    
    coincidence = calculate_coincidence(df)
    
    final_df = grouped.join(meta, how='left')
    final_df = final_df.join(coincidence, how='left').fillna(0)
    
    final_df['peak_overlap'] = final_df.apply(calculate_peak_overlap, axis=1)

    bump_cols = [c for c in final_df.columns if 'pre_n_bumps' in c]
    if bump_cols:
        final_df['total_pre_bumps'] = final_df[bump_cols].sum(axis=1)
        final_df['max_pre_bumps'] = final_df[bump_cols].max(axis=1)
        final_df['std_pre_bumps'] = final_df[bump_cols].std(axis=1)

    post_bump_cols = [c for c in final_df.columns if 'post_n_bumps' in c]
    if post_bump_cols:
        final_df['total_post_bumps'] = final_df[post_bump_cols].sum(axis=1)
        final_df['max_post_bumps'] = final_df[post_bump_cols].max(axis=1)
        final_df['std_post_bumps'] = final_df[post_bump_cols].std(axis=1)

    pairs = [('u', 'g'), ('g', 'r'), ('r', 'i'), ('i', 'z')]
    
    for blue, red in pairs:
        if f"{blue}_max" in final_df.columns and f"{red}_max" in final_df.columns:
            final_df[f'flux_ratio_{blue}_{red}'] = final_df[f"{blue}_max"] / (final_df[f"{red}_max"] + 1.0)

    for blue, red in pairs:
        if f"{blue}_mean" in final_df.columns and f"{red}_mean" in final_df.columns:
            final_df[f'color_{blue}_{red}_mean'] = final_df[f"{blue}_mean"] - final_df[f"{red}_mean"]
        if f"{blue}_min" in final_df.columns and f"{red}_min" in final_df.columns:
            final_df[f'color_{blue}_{red}_min'] = final_df[f"{blue}_min"] - final_df[f"{red}_min"]

    if (
        'u_max' in final_df.columns and
        'r_max' in final_df.columns and
        'u_last_flux' in final_df.columns and
        'r_last_flux' in final_df.columns
    ):
        r_max_safe = final_df['r_max'].replace(0, np.nan)
        r_last_safe = final_df['r_last_flux'].replace(0, np.nan)
        peak_ratio = final_df['u_max'] / (r_max_safe + 1e-5)
        tail_ratio = final_df['u_last_flux'] / (r_last_safe + 1e-5)
        final_df['cooling_ratio'] = (tail_ratio / (peak_ratio + 1e-9)).fillna(0)
        final_df['color_change_u_r'] = (
            (final_df['u_last_flux'] - final_df['r_last_flux']) -
            (final_df['u_max'] - final_df['r_max'])
        )

    rise_cols = [c for c in final_df.columns if 'rise_15' in c]
    fall_cols = [c for c in final_df.columns if 'fall_15' in c]
    if rise_cols and fall_cols:
        total_rise = final_df[rise_cols].sum(axis=1)
        total_fall = final_df[fall_cols].sum(axis=1)
        final_df['rise_fall_ratio'] = total_fall / (total_rise + 1.0)
        final_df['rise_fall_diff'] = total_fall - total_rise

    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    peak_times = [f"{b}_t_peak" for b in bands if f"{b}_t_peak" in final_df.columns]
    if len(peak_times) > 1:
        final_df['peak_time_std'] = final_df[peak_times].std(axis=1)
        final_df['peak_time_range'] = final_df[peak_times].max(axis=1) - final_df[peak_times].min(axis=1)

    amp_cols = [f"{b}_amplitude" for b in bands if f"{b}_amplitude" in final_df.columns]
    if len(amp_cols) > 1:
        final_df['amplitude_std'] = final_df[amp_cols].std(axis=1)
        final_df['amplitude_max_ratio'] = final_df[amp_cols].max(axis=1) / (final_df[amp_cols].mean(axis=1) + 1e-5)

    neumann_cols = [c for c in final_df.columns if '_neumann' in c and 'tail' not in c]
    if neumann_cols:
        final_df['mean_neumann'] = final_df[neumann_cols].mean(axis=1)
        final_df['max_neumann'] = final_df[neumann_cols].max(axis=1)

    sig_cols = [c for c in final_df.columns if 'peak_significance' in c]
    if sig_cols:
        final_df['max_peak_significance'] = final_df[sig_cols].max(axis=1)
        final_df['mean_peak_significance'] = final_df[sig_cols].mean(axis=1)

    n_peaks_cols = [c for c in final_df.columns if '_n_peaks' in c]
    if n_peaks_cols:
        final_df['total_n_peaks'] = final_df[n_peaks_cols].sum(axis=1)

    final_df.to_csv(output_path)

if __name__ == "__main__":
    make_features(TRAIN_FILE, OUTPUT_TRAIN, is_train=True)
    make_features(TEST_FILE, OUTPUT_TEST, is_train=False)
