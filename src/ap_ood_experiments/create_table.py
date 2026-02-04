import json
import os
from pathlib import Path

import hydra
import pandas as pd
import numpy as np


def create_method_mapping(config_methods):
    mapping = dict()
    for method in config_methods:
        mapping[method.target] = method.name
    return mapping
        

def create_dataset_mapping(config_datasets):
    mapping = dict()
    for dataset in config_datasets:
        mapping[dataset.dir_name] = dataset.name
    return mapping


def best_mean_is_bold(mean, std, metric):
    mean = np.array(mean)[0]
    std = np.array(std)[0]
    if metric == 'FPR':
        return np.array(mean) == np.min(mean)
    if metric == 'AUROC':
        return np.array(mean) == np.max(mean)
    if metric == 'Accuracy':
        return np.array(mean) == np.max(mean)

def best_inside_std_div_is_bold(mean, std, metric):
    mean = np.array(mean)[0]
    std = np.array(std)[0]
    if metric == 'FPR':
        best_arg = np.argmin(mean)
        best_mean = mean[best_arg]
        best_std = std[best_arg]
        return (mean <= best_mean + best_std)
    if metric == 'AUROC':
        best_arg = np.argmax(mean)
        best_mean = mean[best_arg]
        best_std = std[best_arg]
        return (mean >= best_mean - best_std)
    if metric == 'Accuracy':
        best_arg = np.argmax(mean)
        best_mean = mean[best_arg]
        best_std = std[best_arg]
        return (mean >= best_mean - best_std)


def compile_row(mean, std, total_mean, metric, method_name, datasets, show_std=True):
    if metric == 'fpr95':
        latex = r'\multirow{-2}{*}{' + method_name + r'}&FPR95 $\downarrow$'
    elif metric == 'auroc':
        latex = r'&AUROC $\uparrow$'
    for dataset in datasets:
        m = mean[mean['dataset'] == dataset]['value'].item()
        s = std[std['dataset'] == dataset]['value'].item()
        b = mean[mean['dataset'] == dataset]['is_bold'].item()
        u = mean[mean['dataset'] == dataset]['is_underline'].item()
        latex += '&$'
        if b:
            latex += r'\mathbf{'
        if u:
            latex += r'\underline{'
        latex += '{:0.2f}'.format(m * 100) 
        if show_std:
            latex += r'^{\pm ' + '{:0.2f}'.format(s * 100) + r'}'
        if b:
            latex += r'}'
        if u:
            latex += r'}'
        latex += '$'
    latex += '&$'
    m = total_mean['value'].item()
    b = total_mean['is_bold'].item()
    u = total_mean['is_underline'].item()
    if b:
        latex += r'\mathbf{'
    if u:
        latex += r'\underline{'
    latex += '{:0.2f}'.format(m * 100)
    if b:
        latex += r'}'
    if u:
        latex += r'}'
    latex += '$'
    latex += r'\\' + '\n'
    return latex

def create_table(input_means, input_stds, total_input_means, output_means, output_stds, total_output_means, method_mapping, show_std=True):
    if input_means is None and output_means is None:
        return 'No results found'
    if input_means is not None:
        datasets = sorted(set(input_means['dataset']))
    elif output_means is not None:
        datasets = sorted(set(output_means['dataset']))
    if input_means is not None and output_means is not None:
        assert datasets == sorted(set(output_means['dataset']))
    header = r'{ll' + ('l' * len(datasets)) + r'l}'
    latex = r'\begin{tabular}' + header + '\n'
    latex += '&&' + '&'.join(['{' + ds + '}' for ds in datasets]) + r'&{Mean}\\' + '\n'
    latex += r'\midrule' + '\n' + r'\multicolumn{' + str(len(datasets) + 2) + r'}{c}{Input OOD}\\' + '\n' + r'\midrule' + '\n'
    if input_means is not None:
        latex += create_data_rows(input_means, input_stds, total_input_means, method_mapping, show_std=show_std)
    latex += r'\midrule' + '\n' + r'\multicolumn{' + str(len(datasets) + 2) + r'}{c}{Output OOD}\\' + '\n' + r'\midrule' + '\n'
    if output_means is not None:
        latex += create_data_rows(output_means, output_stds, total_output_means, method_mapping, show_std=show_std)
    latex += r'\midrule' + '\n'
    # acc_mean = df[(df['metric'] == 'Accuracy-Mean') & (df['dataset'] == 'Acc')][methods]
    # acc_std = df[(df['metric'] == 'Accuracy-Std') & (df['dataset'] == 'Acc')][methods]
    # latex += compile_row(acc_mean, acc_std, 'Accuracy', 'Acc', bold_tester)
    latex += r'\end{tabular}'
    return latex

def create_data_rows(means, stds, total_means, method_mapping, show_std=True):
    latex = ''
    datasets = sorted(set(means['dataset']))
    rowcolors = ['gray!10', 'white'] * 1000
    methods = [method for method in method_mapping.items() if method[0] in set(means['method_qualifier']) and method[1] in set(means['method'])]
    for method, rowcolor in zip(methods, rowcolors):
        #if dataset == 'Mean':
        #    latex += r'\midrule'
        #    rowcolor = 'white'
        for metric in sorted(set(means['metric'])):
            latex += r'\rowcolor{' + rowcolor + r'}' + '\n'
            mean = means[(means['metric'] == metric) & (means['method_qualifier'] == method[0])]#[datasets]
            std = stds[(stds['metric'] == metric) & (stds['method_qualifier'] == method[0])]#[datasets]
            total_mean = total_means[(total_means['metric'] == metric) & (total_means['method_qualifier'] == method[0])]
            latex += compile_row(mean, std, total_mean, metric, method[1], datasets, show_std=show_std)
    return latex


def mark_bold(means):
    
    def lowest_is_bold(df):
        df['is_bold'] = False
        lowest = np.min(df['value_str'].unique().astype(float))
        df.loc[df['value_str'].astype(float) == lowest, 'is_bold'] = True
        #df.loc[lowest, 'is_bold'] = True
        return df
    
    def highest_is_bold(df):
        df['is_bold'] = False
        highest = np.max(df['value_str'].unique().astype(float))
        df.loc[df['value_str'].astype(float) == highest, 'is_bold'] = True
        # df.loc[highest, 'is_bold'] = True
        return df
    
    fpr_95 = means[means['metric'] == 'fpr95'].groupby(['dataset', 'metric']).apply(lowest_is_bold)
    not_fpr_95 = means[means['metric'] != 'fpr95'].groupby(['dataset', 'metric']).apply(highest_is_bold)
    
    return pd.concat([fpr_95, not_fpr_95]).reset_index(drop=True)

def mark_underline(means):

    def second_lowest_is_underline(df):
        df['is_underline'] = False
        if len(df['value_str'].astype(float).unique()) < 2:
            return df
        second_lowest = np.sort(df['value_str'].astype(float).unique())[1]
        if df.loc[df['value_str'].astype(float) == second_lowest, 'is_bold'].all():
            return df
        df.loc[df['value_str'].astype(float) == second_lowest, 'is_underline'] = True
        return df
    
    def second_highest_is_underline(df):
        df['is_underline'] = False
        if len(df['value_str'].astype(float).unique()) < 2:
            return df
        second_highest = np.sort(df['value_str'].astype(float).unique())[::-1][1]
        if df.loc[df['value_str'].astype(float) == second_highest, 'is_bold'].all():
            return df
        df.loc[df['value_str'].astype(float) == second_highest, 'is_underline'] = True
        return df
    
    fpr_95 = means[means['metric'] == 'fpr95'].groupby(['dataset', 'metric'], axis='index').apply(second_lowest_is_underline)
    not_fpr_95 = means[means['metric'] != 'fpr95'].groupby(['dataset', 'metric'], axis='index').apply(second_highest_is_underline)

    return pd.concat([fpr_95, not_fpr_95]).reset_index(drop=True)

def aggregate_results(result_table, do_mark_bold=True, do_mark_underline=True):
    result_means = result_table.drop('file', axis=1).groupby(['metric', 'dataset', 'method_qualifier', 'method']).mean().reset_index()
    result_means['value_str'] = result_means['value'].apply(lambda x: '{:.2f}'.format(x * 100))
    total_means = result_table.drop(['file', 'dataset'], axis=1).groupby(['metric', 'method_qualifier', 'method']).mean().reset_index()
    total_means['value_str'] = total_means['value'].apply(lambda x: '{:.2f}'.format(x * 100))
    total_means['dataset'] = 'Mean'
    if do_mark_bold:
        result_means = mark_bold(result_means)
        total_means = mark_bold(total_means)
    if do_mark_underline:
        result_means = mark_underline(result_means)
        total_means = mark_underline(total_means)
    result_stds = result_table.drop('file', axis=1).groupby(['metric', 'dataset', 'method_qualifier', 'method']).std().reset_index().fillna(0)
    return result_means, result_stds, total_means


def is_relevant_metric(name):
    return name.startswith('_base/auroc') or name.startswith('_base/fpr95') or name.startswith('auroc') or name.startswith('fpr95') or name.startswith('_mean/auroc') or name.startswith('_mean/fpr95')

def collect_results(methods, method_mapping, dataset_mapping):
    result_list = []
    for method in methods:
        if not method.is_dir() or method.name.startswith('.'):
            # there is a multirun.yaml file when executing a multirun via hydra, skip it
            # if started via hydra-submitit, there is a .submitit directory, skip it
            continue
        if method.name in method_mapping:
            method_name = method_mapping[method.name]
        else:
            print(f'{method.name} not found in method mapping. Skipping...') 
            continue
        result_files = list(method.glob('**/results.json'))  # should have multiple result files for different seeds
        #assert len(result_files) > 1  # we need at least two to compute standard deviation
        for result_file in result_files:
            with open(result_file, 'r') as result:
                result = json.load(result)
            for metric_name, metric_result in result.items():
                if not metric_name.endswith('.mean') and is_relevant_metric(metric_name):
                    metric_name, metric_dataset_path = metric_name.split('.')
                    if '/' in metric_name:
                        _, metric_name = metric_name.split('/')
                    metric_dataset_path = Path(metric_dataset_path)
                    dataset_name = dataset_mapping.get(metric_dataset_path.parts[-2])
                    if dataset_name is None:
                        print(f'{metric_dataset_path.parts[-2]} not found in dataset mapping. Skipping...')
                        continue
                    result_list.append({
                        'metric': metric_name,
                        'dataset': dataset_name,
                        'value': metric_result,
                        'method': method_name,
                        'method_qualifier': method.name,
                        'file': str(result_file),
                    })
    return pd.DataFrame(data=result_list)


@hydra.main(config_path='config_create_table', version_base='1.2')
def main(config):
    input_runs_path = Path(config.input_runs_path)
    output_runs_path = Path(config.output_runs_path)
    input_methods = list(input_runs_path.iterdir())
    output_methods = list(output_runs_path.iterdir())
    method_mapping = create_method_mapping(config.methods)
    dataset_mapping = create_dataset_mapping(config.datasets)
    if input_methods:
        results = collect_results(input_methods, method_mapping, dataset_mapping)
        input_means, input_stds, total_input_means = aggregate_results(results)
        if not config.do_underline:
            input_means['is_underline'] = False
            total_input_means['is_underline'] = False
    else:
        input_means = None
        input_stds = None
    if output_methods:
        results = collect_results(output_methods, method_mapping, dataset_mapping)
        output_means, output_stds, total_output_means = aggregate_results(results)
        if not config.do_underline:
            output_means['is_underline'] = False
            total_output_means['is_underline'] = False
    else:
        output_means = None
        output_stds = None
    
    print(create_table(input_means, input_stds, total_input_means, output_means, output_stds, total_output_means, method_mapping=method_mapping, show_std=config.show_std))


if __name__ == '__main__':
    main()
