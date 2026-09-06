#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POSTECH OIBC Solar - GC_atmp3b

Solar irradiance (nins, W/m^2) estimation using:
- LightGBM Tweedie regression
- temporal / weather interactions
- geographical clustering
- approximate solar geometry
- physical / atmospheric proxy features
- 3-seed ensemble
- night-zero + daytime smoothing post-processing

This is a GitHub-friendly refactor of the original Colab experiment.
Competition data is not included.
"""

from __future__ import annotations

import argparse
import gc
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMRegressor
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
N_ESTIMATORS = 20_000
TWEEDIE_POWER = 1.05
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 500
LOG_PERIOD = 200
N_CLUSTERS = 10

SEEDS = [17, 42, 77]

POST_NIGHT_ZERO = True
NIGHT_ELEV_THRESH_DEG = 1.0
GEO_MIN_DAY_RATIO = 0.25
PROXY_DAY_THR = 0.02
POST_DAY_SMOOTH = True
POST_DAY_SMOOTH_WIN = 2
MIN_PRED_THRESHOLD = 0.001

def print_header():
    print("="*70)
    print("🚀 GC_atmp3b — Tweedie + Location + Solar Geometry + Phys/Astro Decomp + SeedEns3")
    print("="*70)
    print(f"  Tweedie Power: {TWEEDIE_POWER}")
    print(f"  N Estimators : {N_ESTIMATORS}")
    print(f"  LearningRate : {LEARNING_RATE}")
    print(f"  EarlyStop    : {EARLY_STOPPING_ROUNDS}")
    print(f"  Clusters     : {N_CLUSTERS}")
    print(f"  Seeds        : {SEEDS}")
    print(f"  Postproc     : NIGHT_ZERO={POST_NIGHT_ZERO} (geo_thr={NIGHT_ELEV_THRESH_DEG}°), "
          f"DAY_SMOOTH(win={POST_DAY_SMOOTH_WIN})={POST_DAY_SMOOTH}, "
          f"GEO_MIN_DAY_RATIO={GEO_MIN_DAY_RATIO:.2f}, PROXY_THR={PROXY_DAY_THR}, "
          f"MIN_PRED_THR={MIN_PRED_THRESHOLD}")

def print_memory_usage(tag=""):
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    print(f"📊 MEM[{tag}]: {mem_gb:.2f} GB")

def optimize_dtypes(df):
    for col in df.columns:
        if col == 'time' or df[col].dtype == object:
            continue
        cmin, cmax = df[col].min(), df[col].max()
        if str(df[col].dtype).startswith('int'):
            if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df

def normalize_missing(df, exclude=('time','pv_id','type')):
    null_tokens = {'', ' ', 'NA', 'NaN', 'NULL', 'None', 'null', 'nan'}
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            df[c] = df[c].replace(list(null_tokens), np.nan)
            df[c] = pd.to_numeric(df[c], errors='coerce')
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

def ensure_naive_time(series):
    if not is_datetime64_any_dtype(series):
        series = pd.to_datetime(series)
    if is_datetime64tz_dtype(series):
        series = series.dt.tz_localize(None)
    return series

# ============================================================================================
# 시간/주기 & 1차 상호작용
# ============================================================================================
def add_time_and_interactions(df):
    df['time'] = ensure_naive_time(df['time'])

    df['hour'] = df['time'].dt.hour.astype(np.int8)
    df['minute'] = df['time'].dt.minute.astype(np.int8)
    df['month'] = df['time'].dt.month.astype(np.int8)
    df['dayofyear'] = df['time'].dt.dayofyear.astype(np.int16)
    df['dayofweek'] = df['time'].dt.dayofweek.astype(np.int8)
    df['day'] = df['time'].dt.day.astype(np.int8)
    df['quarter'] = df['time'].dt.quarter.astype(np.int8)
    df['weekofyear'] = df['time'].dt.isocalendar().week.astype(np.int8)
    df['time_of_day_minutes'] = (df['hour']*60 + df['minute']).astype(np.int16)

    df['hour_sin']  = np.sin(2*np.pi*df['hour']/24).astype(np.float32)
    df['hour_cos']  = np.cos(2*np.pi*df['hour']/24).astype(np.float32)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12).astype(np.float32)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12).astype(np.float32)

    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(np.int8)
    df['sun_elevation_proxy'] = np.maximum(0, np.sin((df['hour']-6)*np.pi/12)).astype(np.float32)

    if 'uv_idx' in df.columns and 'cloud_a' in df.columns:
        df['clearness'] = (100 - df['cloud_a'].fillna(50)).clip(0,100).astype(np.float32)/100
        df['uv_clearness_interaction'] = (df['uv_idx'].fillna(0) * df['clearness']).astype(np.float32)
    if 'cloud_a' in df.columns:
        cloud_norm = df['cloud_a'].fillna(50).clip(0,100)/100
        df['hour_cloud_interaction'] = (df['hour'] * cloud_norm).astype(np.float32)
        df['peak_cloud_penalty'] = (((df['hour']>=10)&(df['hour']<=14)).astype(np.int8) * cloud_norm).astype(np.float32)
    if 'humidity' in df.columns and 'temp_a' in df.columns:
        df['dryness_index'] = ((100 - df['humidity'].fillna(50)) * np.maximum(0, df['temp_a'].fillna(15)) / 100).astype(np.float32)
    if 'uv_idx' in df.columns:
        df['solar_uv_interaction'] = (df['sun_elevation_proxy'] * df['uv_idx'].fillna(0)).astype(np.float32)
        df['solar_uv_squared'] = (df['solar_uv_interaction']**2).astype(np.float32)
    if 'temp_a' in df.columns:
        noon_distance = np.abs(12 - df['hour'])/12
        df['temp_noon_interaction'] = (df['temp_a'].fillna(15) * (1 - noon_distance)).astype(np.float32)
    if {'cloud_a','cloud_b'} <= set(df.columns):
        df['cloud_mean'] = df[['cloud_a','cloud_b']].mean(axis=1).astype(np.float32)
    elif 'cloud_a' in df.columns:
        df['cloud_mean'] = df['cloud_a'].astype(np.float32)
    if 'humidity' in df.columns:
        df['dryness'] = (100.0 - df['humidity']).astype(np.float32)
    if {'temp_max','temp_min'} <= set(df.columns):
        df['temp_range'] = (df['temp_max'] - df['temp_min']).astype(np.float32)
    if {'wind_spd_a','wind_spd_b'} <= set(df.columns):
        df['wind_spd_mean'] = df[['wind_spd_a','wind_spd_b']].mean(axis=1).astype(np.float32)
    if 'uv_idx' in df.columns and 'cloud_mean' in df.columns:
        df['uv_x_clear'] = (df['uv_idx'] * (100.0 - df['cloud_mean'])).astype(np.float32)
    return df

# ============================================================================================
# 위치 Feature Engineering
# ============================================================================================
def add_location_features(df_train, df_test, n_clusters=10):
    print("\n[위치 Feature Engineering]")
    print("="*50)
    df_train = df_train.copy(); df_test = df_test.copy()
    df_train['__is_train__'] = 1
    df_test['__is_train__'] = 0
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    print(f"  coord1 범위: [{df_all['coord1'].min():.2f}, {df_all['coord1'].max():.2f}]")
    print(f"  coord2 범위: [{df_all['coord2'].min():.2f}, {df_all['coord2'].max():.2f}]")

    df_all['coord1_norm'] = (df_all['coord1'] - df_all['coord1'].min()) / (df_all['coord1'].max() - df_all['coord1'].min() + 1e-9)
    df_all['coord2_norm'] = (df_all['coord2'] - df_all['coord2'].min()) / (df_all['coord2'].max() - df_all['coord2'].min() + 1e-9)
    df_all['coord_distance'] = np.sqrt(df_all['coord1']**2 + df_all['coord2']**2).astype(np.float32)
    df_all['coord_angle'] = np.arctan2(df_all['coord2'], df_all['coord1']).astype(np.float32)
    df_all['coord1_x_coord2'] = (df_all['coord1'] * df_all['coord2']).astype(np.float32)
    df_all['coord1_plus_coord2'] = (df_all['coord1'] + df_all['coord2']).astype(np.float32)
    df_all['coord1_minus_coord2'] = (df_all['coord1'] - df_all['coord2']).astype(np.float32)

    # 위도 추정
    est_lat = df_all['coord1'].astype(float).values
    if (np.nanmax(np.abs(est_lat)) < 5.0) or (np.nanstd(est_lat) < 0.5):
        used_lat = np.full_like(est_lat, 35.0, dtype=np.float32)
        print("  ⚠️ coord1이 위도로 애매 → 위도 35° 고정")
    else:
        used_lat = np.clip(est_lat, -66.0, 66.0).astype(np.float32)
        print("  💡 coord1을 위도로 사용(클램핑)")
    df_all['estimated_latitude'] = used_lat
    df_all['latitude_solar_factor'] = np.cos(np.radians(df_all['estimated_latitude'])).astype(np.float32)
    df_all['hour_x_latitude'] = (df_all['hour'] * df_all['estimated_latitude']).astype(np.float32)
    df_all['sun_elev_proxy_x_lat'] = (df_all['sun_elevation_proxy'] * df_all['latitude_solar_factor']).astype(np.float32)

    # 지리적 클러스터링
    print(f"\n  🎯 지리적 클러스터링 ({n_clusters}개)...")
    pv_coords = df_all.groupby('pv_id')[['coord1', 'coord2']].first().reset_index()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pv_coords['location_cluster'] = kmeans.fit_predict(pv_coords[['coord1', 'coord2']])

    df_all = df_all.merge(pv_coords[['pv_id', 'location_cluster']], on='pv_id', how='left')
    df_all['location_cluster'] = df_all['location_cluster'].astype(np.int8)

    # 클러스터 중심까지 거리
    cluster_centers = kmeans.cluster_centers_
    df_all['distance_to_cluster_center'] = 0.0
    for cid in range(n_clusters):
        m = (df_all['location_cluster'] == cid)
        center = cluster_centers[cid]
        d = np.sqrt((df_all.loc[m, 'coord1'] - center[0])**2 + (df_all.loc[m, 'coord2'] - center[1])**2)
        df_all.loc[m, 'distance_to_cluster_center'] = d
    df_all['distance_to_cluster_center'] = df_all['distance_to_cluster_center'].astype(np.float32)

    # 클러스터별 최대 포텐셜
    print(f"\n  ⭐ 클러스터별 최대 일사량 포텐셜(0.98 quantile)...")
    tr_only = df_all[df_all['__is_train__'] == 1]
    cluster_max_map = tr_only.groupby('location_cluster')['nins'].quantile(0.98).to_dict()
    for cid in sorted(cluster_max_map.keys()):
        print(f"    Cluster {cid}: {cluster_max_map[cid]:.1f} W/m²")
    df_all['cluster_max_potential'] = df_all['location_cluster'].map(cluster_max_map).fillna(650).astype(np.float32)
    df_all['hour_max_ratio'] = (df_all['sun_elevation_proxy'] * df_all['cluster_max_potential']).astype(np.float32)
    df_all['cluster_potential_normalized'] = (df_all['cluster_max_potential'] / 750.0).astype(np.float32)

    df_train_new = df_all[df_all['__is_train__'] == 1].drop(columns=['__is_train__']).reset_index(drop=True)
    df_test_new = df_all[df_all['__is_train__'] == 0].drop(columns=['__is_train__']).reset_index(drop=True)
    return df_train_new, df_test_new, pv_coords, cluster_centers

# ============================================================================================
# 천문/물리: 기본 기하
# ============================================================================================
def add_solar_geometry_approx(df):
    df['time'] = ensure_naive_time(df['time'])
    lat = df['estimated_latitude'].astype(float).values
    doy = df['dayofyear'].astype(int).values

    gamma = 2.0*np.pi*(doy-1)/365.0
    dec = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
           - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
           - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma)).astype(np.float32)

    hour = df['hour'].astype(float).values
    minute = df['minute'].astype(float).values
    h_clk = (hour + minute/60.0 - 12.0) * 15.0 * np.pi/180.0  # clock hour angle [rad]

    latr = np.radians(lat)
    sin_elev = np.sin(latr)*np.sin(dec) + np.cos(latr)*np.cos(dec)*np.cos(h_clk)
    elev = np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0))).astype(np.float32)

    df['solar_elev_deg']   = elev
    df['solar_zenith_deg'] = (90.0 - elev).astype(np.float32)
    df['solar_dec_rad']    = dec.astype(np.float32)
    df['hour_angle_clk']   = h_clk.astype(np.float32)
    df['sin_solar_elev']   = np.sin(np.radians(elev)).astype(np.float32)
    df['cos_solar_elev']   = np.cos(np.radians(elev)).astype(np.float32)
    return df

# 태양시 보정(경도/Equation of Time) + H0/Daylength + 정규화 시각각
def add_daylength_and_solar_time_features(df):
    df = df.copy()
    lon_arr = df['coord2'].astype(float).values if 'coord2' in df.columns else np.full(len(df), 127.0)
    if (np.nanmax(np.abs(lon_arr)) < 5.0) or (np.nanstd(lon_arr) < 0.5):
        lon_arr = np.full(len(df), 127.0)
    df['assumed_longitude'] = lon_arr.astype(np.float32)

    phi = np.radians(df['estimated_latitude'].astype(float).values)
    dec = df['solar_dec_rad'].astype(float).values

    cosH0 = (-np.tan(phi) * np.tan(dec))
    cosH0 = np.clip(cosH0, -1.0, 1.0)
    H0 = np.arccos(cosH0)
    df['H0_rad'] = H0.astype(np.float32)
    df['daylength_hours'] = (2*H0 * 180/np.pi) / 15.0

    doy = df['dayofyear'].astype(float).values
    B = 2*np.pi*(doy-81)/364.0
    EoT_min = 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B)

    L_std = np.round(df['assumed_longitude']/15.0)*15.0
    TC_min = 4*(L_std - df['assumed_longitude']) + EoT_min
    solar_minutes = df['time_of_day_minutes'].astype(float).values + TC_min
    solar_hours = (solar_minutes/60.0)
    h_true = (solar_hours - 12.0) * 15.0 * np.pi/180.0
    df['hour_angle_true'] = h_true.astype(np.float32)
    df['delta_hour_angle'] = (df['hour_angle_true'] - df['hour_angle_clk']).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        h_norm = np.where(H0>0, h_true/H0, 0.0)
    h_norm = np.clip(h_norm, -1.5, 1.5)
    df['hour_angle_norm'] = h_norm.astype(np.float32)
    df['midday_shape'] = (1.0 - (h_norm**2)).clip(0, 1.0).astype(np.float32)
    return df

# ============================================================================================
# 물리 피처 (기본) + 분해형 전송
# ============================================================================================
def add_physical_solar_features(df, tilt_deg=30.0):
    if 'solar_elev_deg' not in df.columns:
        return df
    elev = df['solar_elev_deg'].astype(float).clip(-5, 90).values
    zenith = 90.0 - elev
    cos_z = np.cos(np.radians(zenith))
    cos_z = np.clip(cos_z, 0.0, 1.0)
    df['cos_zenith'] = cos_z.astype(np.float32)

    am = np.full_like(cos_z, np.nan, dtype=np.float32)
    m_mask = cos_z > 0
    am[m_mask] = 1.0 / (cos_z[m_mask] + 0.50572*((96.07995 - zenith[m_mask])**-1.6364))
    am = np.where(np.isfinite(am), am, 1.0).astype(np.float32)
    df['air_mass'] = am

    cs = np.zeros_like(cos_z, dtype=np.float32)
    cz_pos = cos_z > 0
    cs[cz_pos] = 1098.0 * cos_z[cz_pos] * np.exp(-0.059 / np.maximum(cos_z[cz_pos], 1e-3))
    df['cs_ghi_haurwitz'] = cs.astype(np.float32)

    if 'cluster_max_potential' in df.columns:
        ref = (df['cluster_max_potential'].astype(float) + 1e-6).values
        df['cs_ghi_norm'] = (cs / ref).astype(np.float32)
    else:
        max_cs = max(cs.max(), 1e-6)
        df['cs_ghi_norm'] = (cs / max_cs).astype(np.float32)

    eff_elev = np.clip(elev + float(tilt_deg), 0.0, 90.0)
    df['tilted_cosine'] = np.sin(np.radians(eff_elev)).astype(np.float32)

    if 'cloud_mean' in df.columns:
        c = df['cloud_mean'].fillna(50).clip(0,100).values/100.0
        trans = (1.0 - c**1.5)
        df['cs_adj_cloud'] = (cs * trans).astype(np.float32)

    c = (df['cloud_mean'].fillna(50)/100.0).values if 'cloud_mean' in df.columns else np.zeros_like(cos_z)
    h = (df['humidity'].fillna(50)/100.0).values if 'humidity' in df.columns else np.full_like(cos_z, 0.5)
    am_safe = np.maximum(am, 1.0)
    trans_cloud = np.exp(-0.7 * (c ** 1.2) * am_safe)
    trans_humid = np.exp(-0.3 * (h ** 1.3))
    atm_trans = (trans_cloud * trans_humid).astype(np.float32)
    df['atm_trans_index'] = atm_trans

    phys_ghi = (cs * atm_trans).astype(np.float32)
    df['phys_ghi_proxy'] = np.clip(phys_ghi, 0.0, 1400.0)
    return df

# Extraterrestrial / optical / TL proxy
def add_extraterrestrial_and_optical(df):
    df = df.copy()
    if 'dayofyear' not in df.columns or 'cos_zenith' not in df.columns:
        return df
    doy = df['dayofyear'].astype(float).values
    E0 = 1.0 + 0.033 * np.cos(2.0*np.pi*doy/365.0)
    I_sc = 1361.0
    I0 = I_sc * E0
    cosz = df['cos_zenith'].astype(float).clip(0,1).values
    I0h = I0 * cosz
    df['E0_factor'] = E0.astype(np.float32)
    df['I0'] = I0.astype(np.float32)
    df['I0h'] = I0h.astype(np.float32)

    eps = 1e-6
    if 'cs_ghi_haurwitz' in df.columns:
        df['k_cs_haurwitz'] = (df['cs_ghi_haurwitz'].astype(float) / (I0h + eps)).clip(0,1.5).astype(np.float32)
    if 'phys_ghi_proxy' in df.columns:
        df['k_phys_proxy'] = (df['phys_ghi_proxy'].astype(float) / (I0h + eps)).clip(0,1.5).astype(np.float32)
    if {'air_mass','phys_ghi_proxy'}.issubset(df.columns):
        am = df['air_mass'].astype(float).replace([np.inf,-np.inf],np.nan).fillna(1.0)
        trans = (df['phys_ghi_proxy'].astype(float)/(I0h + eps)).clip(eps,1.0)
        TL = (-np.log(trans)/(am + eps)).clip(1.0,8.0)
        df['TL_proxy'] = TL.astype(np.float32)

    if 'air_mass' in df.columns:
        am = df['air_mass'].astype(float).replace([np.inf,-np.inf],np.nan).fillna(1.0)
        df['inv_air_mass'] = (1.0/(am+1e-3)).astype(np.float32)
        df['log_air_mass'] = np.log(am+1e-3).astype(np.float32)

    if {'atm_trans_index','cos_zenith'} <= set(df.columns):
        df['atm_trans_elev_weighted'] = (df['atm_trans_index']*df['cos_zenith']).astype(np.float32)
    return df

# 극한 지표
def add_cs_extreme_index(df):
    if {'dayofyear','cos_zenith','cs_ghi_haurwitz'} <= set(df.columns):
        df = df.copy()
        doy = df['dayofyear'].astype(float).values
        E0 = 1.0 + 0.033*np.cos(2.0*np.pi*(doy-3)/365.0)
        I0 = 1361.0 * E0
        cosz = df['cos_zenith'].astype(float).clip(0,1).values
        I0h = I0 * cosz
        ratio = df['cs_ghi_haurwitz'].astype(float).values/(I0h + 1e-6)
        df['cs_extreme_index'] = np.clip(ratio, 0.0, 1.5).astype(np.float32)
    return df

# ======================= 분해형 전달(추가) & 지수들 =======================
def add_transmittance_decomposition(df):
    df = df.copy()
    if 'air_mass' not in df.columns or 'I0h' not in df.columns:
        return df
    am = df['air_mass'].astype(float).replace([np.inf,-np.inf],np.nan).fillna(1.0).values
    am = np.maximum(am, 1.0)

    T_r = np.exp(-0.0903 * (am**0.84) * (1 + am - am**1.01))
    T = df['temp_a'].astype(float).fillna(15.0).values if 'temp_a' in df.columns else np.full(len(df),15.0)
    RH = df['humidity'].astype(float).clip(1,100).fillna(50.0).values if 'humidity' in df.columns else np.full(len(df),50.0)
    w_proxy = 0.14 * (RH/100.0) * np.exp(0.06*T)
    T_w = np.exp(-0.08 * w_proxy * am**0.7)

    wind = df['wind_spd_mean'].astype(float).fillna(0.0).values if 'wind_spd_mean' in df.columns else np.zeros(len(df))
    aod_proxy = (1 - (np.clip(RH,0,100)/100.0)**0.5) * np.exp(-wind/8.0)
    aod_proxy = np.clip(aod_proxy, 0.02, 0.6)
    T_a = np.exp(-aod_proxy * am**0.9)

    df['T_rayleigh'] = T_r.astype(np.float32)
    df['T_h2o'] = T_w.astype(np.float32)
    df['T_aerosol'] = T_a.astype(np.float32)

    ghi_decomp = df['I0h'].astype(float).values * T_r * T_a * T_w
    df['ghi_phys_decomp'] = np.clip(ghi_decomp, 0.0, 1400.0).astype(np.float32)

    eps = 1e-6
    df['kt_decomp'] = (df['ghi_phys_decomp'] / (df['I0h'] + eps)).clip(0,1.5).astype(np.float32)

    df['anisotropy_proxy'] = (df['ghi_phys_decomp'] / (df['I0h'] + eps)).clip(0,1.2).astype(np.float32)
    zen = np.radians(df['solar_zenith_deg'].astype(float).clip(0,89.9).values)
    df['perez_clearness_proxy'] = (df['ghi_phys_decomp'] / (df['cs_ghi_haurwitz'].astype(float).clip(eps))).clip(0,2.0) * (1/np.cos(zen))
    df['perez_clearness_proxy'] = df['perez_clearness_proxy'].replace([np.inf,-np.inf], np.nan).fillna(0.0).clip(0,5.0).astype(np.float32)
    return df

# Boost features (mismatch, clear-sky prob 등)
def add_boost_features(df):
    df = df.copy()
    eps = 1e-6
    if {'uv_idx','sin_solar_elev'} <= set(df.columns):
        df['uv_elev_weighted'] = (df['uv_idx'].fillna(0).astype(float) * df['sin_solar_elev'].clip(lower=0)).astype(np.float32)
    if {'phys_ghi_proxy','cs_ghi_haurwitz'} <= set(df.columns):
        ki = (df['phys_ghi_proxy'].astype(float) / np.clip(df['cs_ghi_haurwitz'].astype(float), eps, None)).clip(0,2.0)
        cloud = df.get('cloud_mean', pd.Series(50, index=df.index)).fillna(50).astype(float)/100.0
        expected = (1.0 - cloud**1.3).clip(0.05,1.0)
        df['cloud_sun_mismatch'] = (ki - expected).astype(np.float32)
    if 'atm_trans_index' in df.columns:
        atm_t = df['atm_trans_index'].astype(float).clip(0.0, 1.5)
        cloud = df.get('cloud_mean', pd.Series(50, index=df.index)).fillna(50).astype(float)/100.0
        TL = df.get('TL_proxy', pd.Series(3.0, index=df.index)).astype(float).clip(1.0, 8.0)
        cs_prob = atm_t * (1.0 - cloud) * np.exp(-0.12 * (TL - 2.0).clip(0))
        df['clear_sky_prob_proxy'] = cs_prob.clip(0.0, 1.0).astype(np.float32)
        df['sunny_mask_proxy'] = (df['clear_sky_prob_proxy'] > 0.6).astype(np.int8)
    return df

# ============================================================================================
# 기타 유틸/포스트프로세싱
# ============================================================================================
def pv_te_from(train_pv: pd.Series, train_y: pd.Series, apply_pv: pd.Series, global_mean: float):
    te_map = pd.DataFrame({'pv_id': train_pv.values, 'y': train_y.values}).groupby('pv_id')['y'].mean()
    return apply_pv.map(te_map).fillna(global_mean).astype(np.float32)

def apply_postprocessing(test_df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
    y = preds.copy().astype(np.float32)
    # 1) 야간 0
    if POST_NIGHT_ZERO:
        if 'solar_elev_deg' in test_df.columns:
            elev = test_df['solar_elev_deg'].astype(float).values
            geo_day = elev > float(NIGHT_ELEV_THRESH_DEG)
            proxy_day = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)
            USE_GEO = geo_day.mean() >= float(GEO_MIN_DAY_RATIO)
            day_mask = geo_day if USE_GEO else proxy_day
            y[~day_mask] = 0.0
        else:
            proxy_day = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)
            y[~proxy_day] = 0.0
    # 2) 낮 스무딩
    if POST_DAY_SMOOTH and 'pv_id' in test_df.columns:
        if 'solar_elev_deg' in test_df.columns:
            elev = test_df['solar_elev_deg'].astype(float).values
            geo_day = elev > float(NIGHT_ELEV_THRESH_DEG)
            proxy_day = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)
            USE_GEO = geo_day.mean() >= float(GEO_MIN_DAY_RATIO)
            day_mask_all = geo_day if USE_GEO else proxy_day
        else:
            day_mask_all = test_df['sun_elevation_proxy'].astype(float).values > float(PROXY_DAY_THR)
        y_series = pd.Series(y)
        pv_series = test_df['pv_id'].astype(str).values
        idx = np.arange(len(y))
        for _, g in pd.Series(idx).groupby(pv_series):
            g = g.values
            tmp = y_series.iloc[g].copy()
            dm = day_mask_all[g]
            tmp[~dm] = np.nan
            sm = tmp.rolling(POST_DAY_SMOOTH_WIN, center=True, min_periods=1).mean()
            y_series.iloc[g] = np.where(dm, sm.fillna(tmp).values, y_series.iloc[g].values)
        y = y_series.values
    # 3) 음수 컷 + 최소값 스냅
    y = np.maximum(0.0, y)
    y[y <= float(MIN_PRED_THRESHOLD)] = 0.0
    return y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POSTECH OIBC Solar irradiance estimation (GC_atmp3b)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.csv, test.csv, submission_sample.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gc_atmp3b_submission.csv"),
        help="Submission output path",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs/gc_atmp3b_artifacts"),
        help="Directory for feature importance and model summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sub_path = data_dir / "submission_sample.csv"

    for path in (train_path, test_path, sub_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print_header()
    print("\nLoading CSVs...")
    print_memory_usage("start")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sub_path)

    train["row_id"] = np.arange(len(train))
    test["row_id"] = np.arange(len(test))

    print(f"train: {len(train):,} rows x {len(train.columns)} columns")
    print(f"test : {len(test):,} rows x {len(test.columns)} columns")
    print_memory_usage("loaded")

    # 'energy' exists only in train in the supplied competition data.
    if "energy" in train.columns and "energy" not in test.columns:
        print("Removing 'energy' because it is not available in test.")
        train = train.drop(columns=["energy"])

    train = optimize_dtypes(train)
    test = optimize_dtypes(test)
    train = normalize_missing(train)
    test = normalize_missing(test)

    # 1) Time / cyclic interactions
    train = add_time_and_interactions(train)
    test = add_time_and_interactions(test)

    # 2) Location features
    train, test, _, _ = add_location_features(
        train, test, n_clusters=N_CLUSTERS
    )

    # 3) Solar geometry
    train = add_solar_geometry_approx(train)
    test = add_solar_geometry_approx(test)

    # 4) Daylength / solar time
    train = add_daylength_and_solar_time_features(train)
    test = add_daylength_and_solar_time_features(test)

    # 5) Physical solar proxies
    train = add_physical_solar_features(train)
    test = add_physical_solar_features(test)

    # 6) Extraterrestrial / optical proxies
    train = add_extraterrestrial_and_optical(train)
    test = add_extraterrestrial_and_optical(test)

    # 7) Clear-sky extreme index
    train = add_cs_extreme_index(train)
    test = add_cs_extreme_index(test)

    # 8) Atmospheric transmittance decomposition
    train = add_transmittance_decomposition(train)
    test = add_transmittance_decomposition(test)

    # 9) Additional proxy features
    train = add_boost_features(train)
    test = add_boost_features(test)

    # Categorical encoding
    le_pv = LabelEncoder()
    le_pv.fit(
        pd.concat(
            [train["pv_id"].astype(str), test["pv_id"].astype(str)],
            ignore_index=True,
        )
    )
    train["pv_id_encoded"] = le_pv.transform(
        train["pv_id"].astype(str)
    ).astype(np.int32)
    test["pv_id_encoded"] = le_pv.transform(
        test["pv_id"].astype(str)
    ).astype(np.int32)

    if "type" in train.columns and "type" in test.columns:
        le_type = LabelEncoder()
        le_type.fit(
            pd.concat(
                [
                    train["type"].fillna("unknown").astype(str),
                    test["type"].fillna("unknown").astype(str),
                ],
                ignore_index=True,
            )
        )
        train["type_encoded"] = le_type.transform(
            train["type"].fillna("unknown").astype(str)
        ).astype(np.int16)
        test["type_encoded"] = le_type.transform(
            test["type"].fillna("unknown").astype(str)
        ).astype(np.int16)

    exclude = ["time", "pv_id", "nins", "energy", "type", "row_id"]
    features_base = sorted(
        (set(train.columns) & set(test.columns)) - set(exclude)
    )

    print(f"\nBase features: {len(features_base)}")

    X_all = train[features_base].copy()
    y_all = train["nins"].clip(lower=0).astype(np.float32)

    # Group-based holdout: validation PV IDs do not overlap with train PV IDs.
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=0.8,
        random_state=42,
    )
    tr_idx, va_idx = next(
        gss.split(X_all, y_all, groups=train["pv_id"])
    )

    X_train = X_all.iloc[tr_idx].copy()
    X_val = X_all.iloc[va_idx].copy()
    y_train = y_all.iloc[tr_idx].copy()
    y_val = y_all.iloc[va_idx].copy()

    pv_train = set(train.iloc[tr_idx]["pv_id"].unique())
    pv_valid = set(train.iloc[va_idx]["pv_id"].unique())
    print(
        f"PV GroupSplit | train={len(pv_train)} "
        f"valid={len(pv_valid)} overlap={len(pv_train & pv_valid)}"
    )

    # PV target encoding
    global_mean = float(y_train.mean())
    X_train["pv_te"] = pv_te_from(
        train.iloc[tr_idx]["pv_id"],
        y_train,
        train.iloc[tr_idx]["pv_id"],
        global_mean,
    )
    X_val["pv_te"] = pv_te_from(
        train.iloc[tr_idx]["pv_id"],
        y_train,
        train.iloc[va_idx]["pv_id"],
        global_mean,
    )

    test_feat = test[features_base].copy()
    test_feat["pv_te"] = pv_te_from(
        train.iloc[tr_idx]["pv_id"],
        y_train,
        test["pv_id"],
        global_mean,
    )

    features = list(X_train.columns)
    print(f"Total features (with pv_te): {len(features)}")

    val_pred_list = []
    test_pred_list = []
    best_iters = []
    fi_list = []

    for seed in SEEDS:
        print("\n" + "=" * 70)
        print(f"Training Tweedie model | seed={seed}")
        print("=" * 70)

        np.random.seed(seed)

        model = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=TWEEDIE_POWER,
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            num_leaves=511,
            max_depth=-1,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=3.0,
            min_child_samples=30,
            min_split_gain=0.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=["l1", "tweedie"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False,
                ),
                lgb.log_evaluation(period=LOG_PERIOD),
            ],
        )

        best_iter = int(model.best_iteration_)
        best_iters.append(best_iter)

        val_pred = np.maximum(
            0.0,
            model.predict(X_val, num_iteration=best_iter),
        )
        test_pred = np.maximum(
            0.0,
            model.predict(test_feat, num_iteration=best_iter),
        )

        val_pred_list.append(val_pred)
        test_pred_list.append(test_pred)
        fi_list.append(model.feature_importances_.astype(float))

        mae_seed = mean_absolute_error(y_val, val_pred)
        rmse_seed = np.sqrt(mean_squared_error(y_val, val_pred))

        print(
            f"seed={seed} | best_iter={best_iter} "
            f"| val_MAE={mae_seed:.6f} "
            f"| val_RMSE={rmse_seed:.6f}"
        )

        del model
        gc.collect()

    # Three-seed mean ensemble
    val_pred_mean = np.mean(val_pred_list, axis=0)
    test_pred_mean = np.mean(test_pred_list, axis=0)

    # Post-process only once after ensembling.
    post_pred_test = apply_postprocessing(test, test_pred_mean)

    val_mae = mean_absolute_error(y_val, val_pred_mean)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred_mean))

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS (SeedEns3 mean)")
    print(f"best_iterations : {best_iters}")
    print(f"val_MAE         : {val_mae:.6f}")
    print(f"val_RMSE        : {val_rmse:.6f}")
    print("=" * 70)

    if fi_list:
        fi_avg = np.mean(np.vstack(fi_list), axis=0)
        importance_df = (
            pd.DataFrame(
                {
                    "feature": features,
                    "importance_mean": fi_avg,
                }
            )
            .sort_values("importance_mean", ascending=False)
        )
        importance_df.to_csv(
            args.artifacts_dir / "feature_importance_seedens3.csv",
            index=False,
        )

    pred_df = (
        pd.DataFrame(
            {
                "row_id": test["row_id"].values,
                "nins": post_pred_test,
            }
        )
        .sort_values("row_id")
    )

    submission_out = submission.copy()
    submission_out["nins"] = pred_df["nins"].values
    submission_out.to_csv(
        args.output,
        index=False,
        float_format="%.5f",
    )

    summary = pd.DataFrame(
        [
            {
                "tweedie_power": TWEEDIE_POWER,
                "n_estimators": N_ESTIMATORS,
                "best_iters": str(best_iters),
                "learning_rate": LEARNING_RATE,
                "early_stopping": EARLY_STOPPING_ROUNDS,
                "val_mae_mean": float(val_mae),
                "val_rmse_mean": float(val_rmse),
                "n_features": len(features),
                "n_clusters": N_CLUSTERS,
                "seeds": str(SEEDS),
                "post_night_zero": POST_NIGHT_ZERO,
                "night_thr_deg": NIGHT_ELEV_THRESH_DEG,
                "geo_min_day_ratio": GEO_MIN_DAY_RATIO,
                "proxy_day_thr": PROXY_DAY_THR,
                "post_day_smooth": POST_DAY_SMOOTH,
                "smooth_win": POST_DAY_SMOOTH_WIN,
                "min_pred_threshold": MIN_PRED_THRESHOLD,
            }
        ]
    )
    summary.to_csv(
        args.artifacts_dir / "model_summary.csv",
        index=False,
    )

    print(f"\nSaved submission: {args.output.resolve()}")
    print_memory_usage("final")


if __name__ == "__main__":
    main()
