import prescribe

if __name__ == '__main__':
    prescribe.prescribe('2021-01-01',
                        '2021-01-31',
                        'pandora/data/all_2020_ips.csv',
                        'covid_xprize/validation/data/uniform_random_costs.csv',
                        'output.csv')
