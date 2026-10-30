import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from geodatasets import get_path

    # -------------------------------------------------------
    # 1. Kartendaten laden
    # -------------------------------------------------------

    world = gpd.read_file(get_path("naturalearth.land"))

    countries = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/50m_cultural/"
        "ne_50m_admin_0_countries.zip"
    )

    selected_names = [
        "United States of America",
        "Spain",
        "Germany",
    ]

    selected = countries[
        countries["NAME"].isin(selected_names)
    ].copy()

    selected = selected.to_crs(world.crs)

    # -------------------------------------------------------
    # 2. Karte auf die drei Länder clippen
    # -------------------------------------------------------

    clip_area = selected

    world_clipped = gpd.clip(world, clip_area)

    # -------------------------------------------------------
    # 3. Städte beziehungsweise Institutionsstandorte
    # -------------------------------------------------------

    locations = [
        {
            "name": "Cleveland Clinic",
            "city": "Cleveland, OH, USA",
            "lon": -81.6219,
            "lat": 41.5035,
        },
        {
            "name": "Johns Hopkins University",
            "city": "Baltimore, MD, USA",
            "lon": -76.5929,
            "lat": 39.2989,
        },
        {
            "name": "New York University",
            "city": "New York, NY, USA",
            "lon": -73.9974,
            "lat": 40.7295,
        },
        {
            "name": "OhioHealth",
            "city": "Columbus, OH, USA",
            "lon": -83.0007,
            "lat": 39.9690,
        },
        {
            "name": "Washington University in St. Louis",
            "city": "St. Louis, MO, USA",
            "lon": -90.3051,
            "lat": 38.6488,
        },
        {
            "name": "University of Rochester Medical Center",
            "city": "Rochester, NY, USA",
            "lon": -77.6238,
            "lat": 43.1232,
        },
        {
            "name": "Cleveland Clinic Lou Ruvo Center for Brain Health",
            "city": "Las Vegas, NV, USA",
            "lon": -115.1698,
            "lat": 36.1578,
        },
        {
            "name": "University Hospital of Giessen and Marburg",
            "city": "Marburg, Germany",
            "lon": 8.7706,
            "lat": 50.8070,
        },
        {
            "name": "Vall d’Hebron University Hospital",
            "city": "Barcelona, Spain",
            "lon": 2.1487,
            "lat": 41.4275,
        },
        {
            "name": "University Hospital Carl Gustav Carus",
            "city": "Dresden, Germany",
            "lon": 13.7890,
            "lat": 51.0540,
        },
    ]

    locations_gdf = gpd.GeoDataFrame(
        locations,
        geometry=[
            Point(location["lon"], location["lat"])
            for location in locations
        ],
        crs="EPSG:4326",
    )

    # In das CRS der Karte transformieren
    locations_gdf = locations_gdf.to_crs(selected.crs)

    # -------------------------------------------------------
    # 4. Karte erstellen
    # -------------------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 9))

    # Geclipptes Land als Hintergrund
    world_clipped.plot(
        ax=ax,
        color="whitesmoke",
        edgecolor="lightgray",
        linewidth=0.5,
    )

    # Länderflächen transparent hervorheben
    selected.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
    )

    # Institutionsstandorte als Punkte
    locations_gdf.plot(
        ax=ax,
        color="darkred",
        edgecolor="black",
        linewidth=0.8,
        markersize=55,
        zorder=5,
    )

    # Beschriftungen
    # for _, row in locations_gdf.iterrows():
    #     point = row.geometry

    #     ax.annotate(
    #         row["name"],
    #         xy=(point.x, point.y),
    #         xytext=(5, 5),
    #         textcoords="offset points",
    #         fontsize=8,
    #         color="black",
    #         bbox=dict(
    #             boxstyle="round,pad=0.2",
    #             facecolor="white",
    #             edgecolor="none",
    #             alpha=0.75,
    #         ),
    #         zorder=6,
    #     )

    ax.set_title(
        "Standorte der beteiligten Einrichtungen in den USA, Spanien und Deutschland",
        fontsize=14,
    )

    ax.set_axis_off()
    plt.tight_layout()

    plt.savefig('map.svg')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
