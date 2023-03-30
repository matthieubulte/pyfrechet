import pandas as pd
import numpy as np

from metric_spaces import MetricData, NetworkCholesky


def _get_weather_data():
    import requests
    response = requests.get('https://archive-api.open-meteo.com/v1/archive?latitude=40.71&longitude=-74.01&start_date=2016-01-01&end_date=2016-01-31&hourly=temperature_2m,relativehumidity_2m,surface_pressure,precipitation&temperature_unit=fahrenheit&windspeed_unit=mph&precipitation_unit=inch')
    data = response.json()
    weather_df = pd.DataFrame(data['hourly'])
    weather_df.time = pd.to_datetime(weather_df.time)
    weather_df.set_index('time', inplace=True)
    weather_df.rename({
        'temperature_2m': 'temperature',
        'relativehumidity_2m': 'humidity',
        'surface_pressure': 'pressure',
    }, axis=1, inplace=True)
    return weather_df


def _prepare_nyc_taxi(yellow_taxi_path, outpath):
    # This comes from Table 5 in the Supplementatry material of Variable Selection for Global Fre ́chet Regression
    # Missing: Hell's kitchen, Tudor, Medical City, Fort George, Noho, ABD Park, Bowery, Southern tip, White Hall, Tribecca, Wall Street
    zone_to_pickupname = [
        ['Inwood', 'Fort George', 'Washington Heights North', 'Washington Heights South', 'Hamilton Heights', "East Harlem North","East Harlem South", "Central Harlem","Central Harlem North"],
        ['Upper West Side North', 'Upper West Side South', 'Morningside Heights', 'Central Park'],
        ['Yorkville East', 'Yorkville West', 'Lenox Hill East', 'Lenox Hill West', 'Upper East Side North', 'Upper East Side South'],
        ['Lincoln Square East', 'Lincoln Square West', 'Clinton East', 'Clinton West', 'East Chelsea', 'West Chelsea/Hudson Yards', 'Hell’s Kitchen'],
        ['Garment District', 'Times Sq/Theatre District'],
        ['Midtown North', 'Midtown Center'],
        ['Midtown South'],
        ['UN/Turtle Bay South', 'Sutton Place/Turtle Bay North', 'Murray Hill', 'Murray Hill-Queens' 'Kips Bay', 'Gramercy', 'Tudor', 'Medical City', 'Stuy Town/Peter Cooper Village'],
        ['Meat packing district', 'Greenwich Village North', 'Greenwich Village South', 'Meatpacking/West Village West', 'West Village', 'SoHo', 'Little Italy/NoLiTa', 'Chinatown', 'TriBeCa/Civic Center', 'Noho'],
        ['Lower East Side', 'East Village', 'ABD Park', 'Bowery', 'Two Bridges/Seward Park', 'Southern tip', 'White Hall', 'Tribecca', 'Wall Street']
    ]

    # This maps the official zone names from the taxi dataset to the zones defines above
    pickupname_to_zone = dict([(loc, i)  for i in range(10) for loc in zone_to_pickupname[i] ])
    pickupid_to_pickupname = [(i+1, name) for (i, name) in enumerate(["Newark Airport","Jamaica Bay","Allerton/Pelham Gardens","Alphabet City","Arden Heights","Arrochar/Fort Wadsworth","Astoria","Astoria Park","Auburndale","Baisley Park","Bath Beach","Battery Park","Battery Park City","Bay Ridge","Bay Terrace/Fort Totten","Bayside","Bedford","Bedford Park","Bellerose","Belmont","Bensonhurst East","Bensonhurst West","Bloomfield/Emerson Hill","Bloomingdale","Boerum Hill","Borough Park","Breezy Point/Fort Tilden/Riis Beach","Briarwood/Jamaica Hills","Brighton Beach","Broad Channel","Bronx Park","Bronxdale","Brooklyn Heights","Brooklyn Navy Yard","Brownsville","Bushwick North","Bushwick South","Cambria Heights","Canarsie","Carroll Gardens","Central Harlem","Central Harlem North","Central Park","Charleston/Tottenville","Chinatown","City Island","Claremont/Bathgate","Clinton East","Clinton Hill","Clinton West","Co-Op City","Cobble Hill","College Point","Columbia Street","Coney Island","Corona","Corona","Country Club","Crotona Park","Crotona Park East","Crown Heights North","Crown Heights South","Cypress Hills","Douglaston","Downtown Brooklyn/MetroTech","DUMBO/Vinegar Hill","Dyker Heights","East Chelsea","East Concourse/Concourse Village","East Elmhurst","East Flatbush/Farragut","East Flatbush/Remsen Village","East Flushing","East Harlem North","East Harlem South","East New York","East New York/Pennsylvania Avenue","East Tremont","East Village","East Williamsburg","Eastchester","Elmhurst","Elmhurst/Maspeth","Eltingville/Annadale/Prince's Bay","Erasmus","Far Rockaway","Financial District North","Financial District South","Flatbush/Ditmas Park","Flatiron","Flatlands","Flushing","Flushing Meadows-Corona Park","Fordham South","Forest Hills","Forest Park/Highland Park","Fort Greene","Fresh Meadows","Freshkills Park","Garment District","Glen Oaks","Glendale","Governor's Island/Ellis Island/Liberty Island","Governor's Island/Ellis Island/Liberty Island","Governor's Island/Ellis Island/Liberty Island","Gowanus","Gramercy","Gravesend","Great Kills","Great Kills Park","Green-Wood Cemetery","Greenpoint","Greenwich Village North","Greenwich Village South","Grymes Hill/Clifton","Hamilton Heights","Hammels/Arverne","Heartland Village/Todt Hill","Highbridge","Highbridge Park","Hillcrest/Pomonok","Hollis","Homecrest","Howard Beach","Hudson Sq","Hunts Point","Inwood","Inwood Hill Park","Jackson Heights","Jamaica","Jamaica Estates","JFK Airport","Kensington","Kew Gardens","Kew Gardens Hills","Kingsbridge Heights","Kips Bay","LaGuardia Airport","Laurelton","Lenox Hill East","Lenox Hill West","Lincoln Square East","Lincoln Square West","Little Italy/NoLiTa","Long Island City/Hunters Point","Long Island City/Queens Plaza","Longwood","Lower East Side","Madison","Manhattan Beach","Manhattan Valley","Manhattanville","Marble Hill","Marine Park/Floyd Bennett Field","Marine Park/Mill Basin","Mariners Harbor","Maspeth","Meatpacking/West Village West","Melrose South","Middle Village","Midtown Center","Midtown East","Midtown North","Midtown South","Midwood","Morningside Heights","Morrisania/Melrose","Mott Haven/Port Morris","Mount Hope","Murray Hill","Murray Hill-Queens","New Dorp/Midland Beach","North Corona","Norwood","Oakland Gardens","Oakwood","Ocean Hill","Ocean Parkway South","Old Astoria","Ozone Park","Park Slope","Parkchester","Pelham Bay","Pelham Bay Park","Pelham Parkway","Penn Station/Madison Sq West","Port Richmond","Prospect-Lefferts Gardens","Prospect Heights","Prospect Park","Queens Village","Queensboro Hill","Queensbridge/Ravenswood","Randalls Island","Red Hook","Rego Park","Richmond Hill","Ridgewood","Rikers Island","Riverdale/North Riverdale/Fieldston","Rockaway Park","Roosevelt Island","Rosedale","Rossville/Woodrow","Saint Albans","Saint George/New Brighton","Saint Michaels Cemetery/Woodside","Schuylerville/Edgewater Park","Seaport","Sheepshead Bay","SoHo","Soundview/Bruckner","Soundview/Castle Hill","South Beach/Dongan Hills","South Jamaica","South Ozone Park","South Williamsburg","Springfield Gardens North","Springfield Gardens South","Spuyten Duyvil/Kingsbridge","Stapleton","Starrett City","Steinway","Stuy Town/Peter Cooper Village","Stuyvesant Heights","Sunnyside","Sunset Park East","Sunset Park West","Sutton Place/Turtle Bay North","Times Sq/Theatre District","TriBeCa/Civic Center","Two Bridges/Seward Park","UN/Turtle Bay South","Union Sq","University Heights/Morris Heights","Upper East Side North","Upper East Side South","Upper West Side North","Upper West Side South","Van Cortlandt Park", "Van Cortlandt Village","Van Nest/Morris Park","Washington Heights North","Washington Heights South","West Brighton","West Chelsea/Hudson Yards","West Concourse","West Farms/Bronx River","West Village","Westchester Village/Unionport","Westerleigh", "Whitestone","Willets Point","Williamsbridge/Olinville","Williamsburg (North Side)","Williamsburg (South Side)","Windsor Terrace","Woodhaven","Woodlawn/Wakefield","Woodside","World Trade Center","Yorkville East","Yorkville West"])]

    pickupid_to_zone = dict([
        (pid, pickupname_to_zone[name])
        for (pid, name) in pickupid_to_pickupname
        if name in pickupname_to_zone
    ])

    df = pd.read_parquet(yellow_taxi_path)
    df['origin_zone'] = df.PULocationID.map(pickupid_to_zone)
    df['destination_zone'] = df.DOLocationID.map(pickupid_to_zone)
    df['pickup_dt_uptohour'] = df.tpep_pickup_datetime.dt.round('h')
    subdf = df.loc[~(df.origin_zone.isna() | df.destination_zone.isna())].copy()
    def build_counts(grp_df):
        counts = np.zeros((10,10))
        for _, row in grp_df.iterrows():
            o = int(row['origin_zone'])
            d = int(row['destination_zone'])
            counts[o, d] += 1
        return counts.flatten().tolist()

    counts = pd.DataFrame(
        subdf.groupby(by='pickup_dt_uptohour').apply(build_counts),
        columns=['count_matrix']
    )

    hourly_taxi_df = subdf.groupby(by='pickup_dt_uptohour').agg({
        'passenger_count': 'mean',
        'trip_distance': 'mean',
        'fare_amount': 'mean',
        'tip_amount': 'mean',
        'late_hour': 'mean', # this col is constant
        'payment_credit_card': 'sum',
        'payment_cash': 'sum',
        'payment_free': 'sum',
        'payment_dispute': 'sum',
    })
        
    weather_df = _get_weather_data()

    final_df = weather_df.join(hourly_taxi_df).join(pd.DataFrame(counts, columns=['count_matrix']))
    final_df.to_parquet(outpath)


def load_nyc_taxi(path):
    df = pd.read_parquet(path)
    X = df[['temperature', 'humidity', 'pressure', 'precipitation', 'passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'late_hour', 'payment_credit_card', 'payment_cash', 'payment_free', 'payment_dispute']]
    y = MetricData(NetworkCholesky(10), np.c_[[ M for M in df.count_matrix.apply(lambda r: r.reshape((10,10))).values]])
    return X, y
