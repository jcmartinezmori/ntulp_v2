import asyncio
import fitz
import folium
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import playwright
import pickle
import shutil
from src.config import *
import src.helper as helper
import subprocess
from pathlib import Path
from playwright.async_api import async_playwright
from shapely import wkt


def plot_frames(n, objective, timeLimit, epsLimit, iterCountStart, iterCountEnd):

    modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)

    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    os.makedirs(html_dir)

    g, _, lines_df, trips_df = helper.preprocess_load()
    samples_df = pd.read_csv('{0}/results/instances/samples_df_{1}_{2}.csv'.format(RELPATH, FILENAME, n))

    areas_df = gpd.read_file('./data/community_areas.csv')
    areas_df["geometry"] = areas_df["the_geom"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(areas_df, geometry="geometry", crs="EPSG:4326")

    for iterCount in range(iterCountStart, iterCountEnd + 1):

        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                RELPATH, FILENAME, modelname, iterCount), 'rb'
        ) as file:
            x_N, _, _, _, _, S, _, _ = pickle.load(file)

        # lines_df['width'] = [LINESCALING * np.log2(1 + x_N[j] / lines_df.iloc[j].length) for j in range(len(x_N))]
        lines_df['width'] = [1 / (2 * LINESCALING) * np.log2(1 + x_N[j]) for j in range(len(x_N))]
        # lines_df['width'] = [x_N[j] / LINESCALING for j in range(len(x_N))]

        for _, data in g.nodes(data=True):
            data['sample_ct'] = 0
        if S is not None:
            for _, sample in samples_df.iterrows():
                if sample.community_area in S:
                    g.nodes[int(sample.node)]['sample_ct'] = int(sample.ct)

        folium_map = folium.Map(location=CENTER, zoom_start=11, tiles=None)
        folium.TileLayer('OpenStreetMap', opacity=1/6).add_to(folium_map)
        for _, line in lines_df.iterrows():
            folium.PolyLine(
                line.coords, color=line.hexcolor, weight=line.width, opacity=1, tooltip=line.name
            ).add_to(folium_map)
        folium_map.save('{0}/pre1_{1}_{2}_{3}.html'.format(html_dir, FILENAME, modelname, iterCount))
        for u, data in g.nodes(data=True):
            if data['sample_ct']:
                folium.CircleMarker(
                    location=(data['y'], data['x']), color=HEXBLACK, radius=np.log2(1 + data['sample_ct']), weight=0,
                    fill=True, fill_opacity=1, tooltip=u
                ).add_to(folium_map)

        if S is not None:
            S_gdf = gdf.loc[gdf.index.intersection(S)]
            folium.GeoJson(
                S_gdf.__geo_interface__,
                style_function=lambda feat: {
                    "color": HEXBLACK,
                    "weight": 2,
                    "fillColor": HEXGOLD,
                    "fillOpacity": 0.25,
                }
            ).add_to(folium_map)

        folium_map.save('{0}/pos1_{1}_{2}_{3}.html'.format(html_dir, FILENAME, modelname, iterCount))
        folium_map.save('{0}/pos2_{1}_{2}_{3}.html'.format(html_dir, FILENAME, modelname, iterCount))


async def convert_html_to_images(html_dir, pdf_dir, pdf_crop_dir, png_crop_dir, cropbox, png_zoom, objective=None):

    if os.path.exists(pdf_dir):
        shutil.rmtree(pdf_dir)
    os.makedirs(pdf_dir)
    if os.path.exists(pdf_crop_dir):
        shutil.rmtree(pdf_crop_dir)
    os.makedirs(pdf_crop_dir)
    if os.path.exists(png_crop_dir):
        shutil.rmtree(png_crop_dir)
    os.makedirs(png_crop_dir)

    html_files = sorted(Path(html_dir).glob('*.html'), key=lambda f: f.stat().st_ctime)

    async with async_playwright() as p:

        browser = await p.chromium.launch()
        page = await browser.new_page()

        for i, html_file in enumerate(html_files):

            file_url = html_file.resolve().as_uri()
            if objective is None:
                pdf_out = Path(pdf_dir)/f'frame_{i:04d}.pdf'
            else:
                pdf_out = Path(pdf_dir) / f'{objective}_frame_{i:04d}.pdf'

            await page.goto(file_url)
            await page.set_viewport_size({'width': 1920, 'height': 1080})
            await page.wait_for_load_state('networkidle')
            await page.pdf(path=pdf_out, width='1920', height='1080', print_background=False)

        await browser.close()

    pdf_files = sorted(Path(pdf_dir).glob('*.pdf'), key=lambda f: f.stat().st_ctime)
    for pdf_file in pdf_files:

        pdf_crop_out = Path(pdf_crop_dir)/pdf_file.name
        png_crop_out = Path(png_crop_dir)/pdf_file.name.replace('.pdf', '.png')

        doc = fitz.open(pdf_file)
        page = doc[0]
        page.set_cropbox(fitz.Rect(cropbox))
        doc.save(pdf_crop_out)

        pix = page.get_pixmap(matrix=fitz.Matrix(png_zoom, png_zoom))
        pix.save(png_crop_out)
        doc.close()


if __name__ == '__main__':
    n = 'community-area'
    objective = 'utilitarian'
    timeLimit = 300
    epsLimit = 1
    iterCountStart = -1
    iterCountEnd = 100
    html_dir = './results/frames/{0}/html'.format(objective)
    pdf_dir = './results/frames/{0}/pdf'.format(objective)
    pdf_crop_dir = './results/frames/{0}/pdf_crop'.format(objective)
    png_crop_dir = './results/frames/{0}/png_crop'.format(objective)
    cropbox = (460, 180, 810, 710)
    png_zoom = 4
    try:
        plot_frames(n, objective, timeLimit, epsLimit, iterCountStart, iterCountEnd)
        asyncio.run(convert_html_to_images(html_dir, pdf_dir, pdf_crop_dir, png_crop_dir, cropbox, png_zoom, objective=objective))
    except FileNotFoundError:
        pass
    """
    ffmpeg -framerate 15 -i results/frames/png_crop/frame_%04d.png -crf 18 -preset slow -pix_fmt yuv420p results/frames/frames.mp4
    """
