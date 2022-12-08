from iz_val import read_pill_data

def get_spl():
    df = read_pill_data()
    c3pi_spl = df.loc[df['Layout'] == 'MC_SPL_SPLIMAGE_V3.0']
    return c3pi_spl