'''
Descripttion: 
version: 
Author: Yang Zhong & Shixu Liu
Date: 2023-04-24 22:22:09
LastEditors: Yang Zhong
LastEditTime: 2024-05-16 21:47:08
'''
import argparse
import yaml
from mpi4py import MPI
from .EPC_calculator import EPC_calculator

def main(config_file_name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    rank_size = comm.Get_size()

    with open(config_file_name, encoding='utf-8') as rstream:
        input_data = yaml.load(rstream, yaml.SafeLoader)

    epc_cal = EPC_calculator(input_data, comm)
    epc_cal.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EPC Calculator with a specified configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    
    args = parser.parse_args()
    
    main(args.config)

