
# --- Load model ML
model_path = './models/gradientboosting.pkl'
stacking_model = joblib.load(model_path)

# --- Load scaler
scaler_path = './models/standard_scaler.pkl'
scaler = joblib.load(scaler_path)


# --- Fungsi Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c

# --- Fungsi jarak jalan via OSRM
def get_road_distance(lat1, lon1, lat2, lon2):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        response = requests.get(url)
        data = response.json()
        if "routes" in data and len(data["routes"]) > 0:
            distance = data["routes"][0]["distance"]
            geometry = data["routes"][0]["geometry"]
            return distance, geometry
        return None, None
    except:
        return None, None

# --- Hitung nearby customers
def count_nearby_customers(customers, odp_lat, odp_lon, radius=250):
    return sum(
        haversine(odp_lat, odp_lon, row['latitude'], row['longitude']) <= radius
        for _, row in customers.iterrows()
    )

# --- Normalisasi
def norm(col, smaller_better=True):
    min_val, max_val = col.min(), col.max()
    if max_val - min_val == 0:
        return col * 0 + 0.5
    return 1 - (col - min_val) / (max_val - min_val) if smaller_better else (col - min_val) / (max_val - min_val)

# --- Load data
customers = pd.read_excel('/data/customers.xlsx')
odp = pd.read_excel('/data/odp.xlsx')
poi = pd.read_excel('/data/poi.xlsx')

# --- Input user
user_name = input("Masukkan Nama user: ")
user_lat = float(input("Masukkan Latitude user: "))
user_lon = float(input("Masukkan Longitude user: "))

# --- Hitung distance_haversine semua ODP
odp['distance_haversine'] = odp.apply(lambda r: haversine(user_lat, user_lon, r['latitude'], r['longitude']), axis=1)

# --- Filter radius 250m
candidates = odp[odp['distance_haversine'] <= 250].copy()
if candidates.empty:
    print("⚠️ Tidak ada ODP dalam radius 250 m. Selesai.")
else:
    # --- Hitung distance_road & geometry
    road_distances, geometries = [], []
    for _, row in candidates.iterrows():
        d_road, geom = get_road_distance(user_lat, user_lon, row['latitude'], row['longitude'])
        road_distances.append(d_road or row['distance_haversine'] * 1.3)
        geometries.append(geom)
    candidates['distance_road'] = road_distances
    candidates['geometry'] = geometries

    # --- Utilization ratio & kategori
    candidates['utilization_ratio'] = (candidates['USED'] + candidates['RSV']) / candidates['IS_TOTAL']
    color_map = {"HITAM": 0, "MERAH": 1, "KUNING": 2, "HIJAU": 3}
    candidates['kategori_encoded'] = candidates['Kategori'].map(color_map)

    # --- Filter kapasitas <100% dan bukan HITAM
    candidates = candidates[(candidates['utilization_ratio'] < 1.0) & (candidates['Kategori'] != 'HITAM')]
    if candidates.empty:
        print("⚠️ Semua ODP penuh atau HITAM. Selesai.")
    else:
        # --- Hitung nearby_customers
        candidates['nearby_customers'] = candidates.apply(
            lambda r: count_nearby_customers(customers, r['latitude'], r['longitude']), axis=1
        )

        # --- Road distance category
        def categorize_road_distance(d):
            if d <= 50: return 5
            elif d <= 100: return 4
            elif d <= 150: return 3
            elif d <= 200: return 2
            elif d <= 250: return 1
            else: return 0

        candidates['road_distance_category'] = candidates['distance_road'].apply(categorize_road_distance)

        # --- Normalisasi & scoring
        candidates['distance_road_norm'] = norm(candidates['distance_road'], True)
        candidates['distance_haversine_norm'] = norm(candidates['distance_haversine'], True)
        candidates['utilization_ratio_norm'] = norm(candidates['utilization_ratio'], True)
        candidates['nearby_customers_norm'] = norm(candidates['nearby_customers'], False)
        candidates['kategori_encoded_norm'] = norm(candidates['kategori_encoded'], False)

        # --- (Distance-first 5W): weights as agreed
        w_distance_road = 0.45
        w_haversine = 0.15
        w_util = 0.20
        w_nearby = 0.10
        w_kategori = 0.10

        candidates['score'] = (
            w_distance_road * candidates['distance_road_norm'] +
            w_haversine * candidates['distance_haversine_norm'] +
            w_util * candidates['utilization_ratio_norm'] +
            w_nearby * candidates['nearby_customers_norm'] +
            w_kategori * candidates['kategori_encoded_norm']
        )

        # --- Prediksi ML (dengan scaler)
        model_features = [
            'distance_haversine', 'distance_road', 'road_distance_category',
            'utilization_ratio', 'kategori_encoded', 'nearby_customers', 'score'
        ]
        X_candidates = candidates[model_features]

        # Skala fitur dan buat ulang DataFrame agar nama kolom tetap ada
        X_candidates_scaled = scaler.transform(X_candidates)
        X_candidates_scaled_df = pd.DataFrame(X_candidates_scaled, columns=model_features)

        # Prediksi probabilitas dengan nama fitur yang valid (hilang warning)
        candidates['choose_prob'] = stacking_model.predict_proba(X_candidates_scaled_df)[:, 1]

        # --- Gabungkan score & ML untuk prioritas
        alpha, beta = 0.6, 0.4
        candidates['combined_score'] = alpha * candidates['score'] + beta * candidates['choose_prob']

        # --- Pilih ODP terbaik
        best_idx = candidates['combined_score'].idxmax()
        candidates['choose'] = 0
        candidates.loc[best_idx, 'choose'] = 1

        # --- Tampilkan hasil terminal
        display_cols = [
            'nama', 'distance_haversine', 'distance_road', 'road_distance_category',
            'utilization_ratio', 'kategori_encoded', 'nearby_customers',
            'score', 'choose_prob', 'combined_score', 'choose'
        ]
        print(f"\n=== ODP Kandidat untuk {user_name} ===")
        print(candidates[display_cols].sort_values(by='combined_score', ascending=False).reset_index(drop=True))

        # --- Peta Folium
        odp_in_circle = candidates[candidates['distance_haversine'] <= 250].copy()
        m = folium.Map(location=[user_lat, user_lon], zoom_start=17)

        # --- Lingkaran Haversine ungu
        folium.Circle(
            location=[user_lat, user_lon],
            radius=250,
            color="purple",
            fill=True,
            fill_opacity=0.1,
            popup=f"<b style='font-size:14px;'>Radius 250m</b><br>{len(odp_in_circle)} ODP ditemukan"
        ).add_to(m)

        # --- Marker user
        folium.Marker(
            location=[user_lat, user_lon],
            popup=f"<b style='font-size:16px;'>User: {user_name}</b>",
            icon=folium.Icon(color="purple", icon="home")
        ).add_to(m)

       # --- Marker ODP + rute
route_colors = ["blue", "green", "red", "orange", "darkred", "cadetblue", "purple"]
legend_html = """
<div style="
    position: fixed; top: 20px; right: 20px; width: 260px;
    background: white; border: 2px solid black; z-index:9999;
    font-size:15px; padding:10px; border-radius:10px;
    box-shadow:2px 2px 8px gray;">
<b>Icons di Peta:</b><br>
<i class="fa fa-home fa-lg" style="color:purple;"></i> <b>User Baru</b><br>
<i class="fa fa-user fa-lg" style="color:blue;"></i> <b>Pelanggan</b><br>
<i class="fa fa-star fa-lg" style="color:orange;"></i> <b>POI</b><br>
<i class="fa fa-cloud fa-lg" style="color:green;"></i> <b>ODP Terpilih</b><br>
<i class="fa fa-cloud fa-lg" style="color:red;"></i> <b>ODP Tidak Terpilih</b><br>
<hr style='margin:6px 0;'>
<b>Rute Kabel Fiber Optik ke ODP:</b><br>
"""

for i, (_, row) in enumerate(odp_in_circle.iterrows()):
    color = 'green' if row['choose'] == 1 else 'red'
    popup_html = (
        f"<div style='font-size:15px; line-height:1.5;'>"
        f"<b>ODP {row['nama']}</b><br>"
        f"<b>Combined Score:</b> {row['combined_score']:.3f}<br>"
        f"Nearby: {row['nearby_customers']} pelanggan<br>"
        f"Utilization: {row['utilization_ratio']*100:.1f}%<br>"
        f"Kategori: {row['Kategori']}"
        f"</div>"
    )
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=popup_html,
        icon=folium.Icon(color=color, icon='cloud')
    ).add_to(m)

    # Rute jika tersedia
    if row['geometry'] is not None:
        coords = [(coord[1], coord[0]) for coord in row['geometry']['coordinates']]
        route_color = route_colors[i % len(route_colors)]
        folium.PolyLine(coords, color=route_color, weight=4, opacity=0.7,
                        popup=f"<b style='font-size:14px;'>Rute Kabel Fiber Optik ke {row['nama']}</b>").add_to(m)
        legend_html += f"<i style='background:{route_color};width:16px;height:16px;display:inline-block;'></i>&nbsp;{row['nama']}<br>"

# --- Marker pelanggan
for _, row in customers.iterrows():
    folium.Marker([row['latitude'], row['longitude']],
                  popup=f"<b style='font-size:14px;'>{row.get('nama', 'Customer')}</b>",
                  icon=folium.Icon(color='blue', icon='user')).add_to(m)

# --- Marker POI
for _, row in poi.iterrows():
    folium.Marker([row['latitude'], row['longitude']],
                  popup=f"<b style='font-size:14px;'>{row.get('nama', 'POI')}</b>",
                  icon=folium.Icon(color='orange', icon='star')).add_to(m)

legend_html += "</div>"
m.get_root().html.add_child(folium.Element(legend_html))
display(m)
