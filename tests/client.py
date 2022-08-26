import time
from eden.client import Client
from eden.datatypes import Image

## set up a client
c = Client(url="http://0.0.0.0:5656", username="abraham")

## define input args to be sent
config = {
    "mode": "interrogate",
    "image_url": "https://cdn.discordapp.com/attachments/1006144058940469268/1011746484418531418/a_futuristic_city_with_neon_lights_and_buildings_a_3D_render_by_Victor_Mosquera_featured_on_polycount_cubo-futurism_rendered_in_cinema4d_rendered_in_unreal_engine_octane_render_v1.png"
}

config = {
    "mode": "interrogate",
    "image_url": "https://cdn.discordapp.com/attachments/1006144058940469268/1011744967653326898/a_painting_of_a_mountain_with_a_lake_below_it_an_art_deco_painting_by_Lawren_Harris_trending_on_behance_crystal_cubism_fauvism_cubism_chillwave_v4.png"
}

# start the task
run_response = c.run(config)

# check status of the task, returns the output too if the task is complete
results = c.fetch(token=run_response["token"])
print(results)

# one eternity later
time.sleep(2)

while True:
    ## check status again, hopefully the task is complete by now
    results = c.fetch(token=run_response["token"])
    print(results)
    time.sleep(5)