import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import gzip
import re
import itertools
import zipfile
import tempfile
import os
import string
import random
from streamlit.components.v1 import html
import io
from sklearn.metrics import r2_score
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
import matplotlib
import zipfile
import struct
import seaborn as sns
import csv
import shutil
from scipy.signal import savgol_filter, find_peaks

# Function to map react_id to well
def map_react_id_to_well(react_id):
    row = chr(65 + (react_id - 1) // 24)
    col = (react_id - 1) % 24 + 1
    return f'{row}{col:02d}'

def extract_max_cycles(protocol_content):
    # Look for 'PLATEREAD', then any characters until 'GOTO', then an integer followed by a comma, and then capture the cycle number
    pattern = r'PLATEREAD.*?GOTO \d+,(\d+)'
    match = re.search(pattern, protocol_content)
    
    if match:
        max_cycles = int(match.group(1)) + 1
        return max_cycles
    else:
        # Default value or handle the absence of a matching pattern as needed
        return 40

def extract_data_from_zpcr(temp_dir, pcr_data_basename):
    data = []
    melt_data = []
    dye_id = 'FAM'  # Assuming 'channel 1' corresponds to FAM dye

    # Extract temperature data from the .alf file
    alf_file_path = os.path.join(temp_dir, f'{pcr_data_basename}.alf')  

    alf_file_path = os.path.join(temp_dir, f'{pcr_data_basename}.alf')

    temperatures = {}
    previous_line = ''

    # Open and read the file line by line
    cycle_no = 1
    with open(alf_file_path, 'r') as file:
        for line in file:
            # Check if the current line contains 'Plate Read'
            if 'Plate Read' in line:
                # Extract the temperature from the previous line
                parts = previous_line.split('*')
                try:
                    # Assuming the temperature is always at the 5th position
                    temperature = float(parts[4])
                    temperatures[cycle_no] = temperature
                    cycle_no += 1
                except IndexError:
                    # If the expected position is out of range, log an error or skip
                    print("Error extracting temperature from line:", previous_line)
                    continue
            # Update the previous line
            previous_line = line
    
    # First, read the ProtocolRunDefinition.txt to find the number of cycles
    protocol_file_path = os.path.join(temp_dir, 'ProtocolRunDefinition.txt')
    if os.path.exists(protocol_file_path):
        with open(protocol_file_path, 'r') as protocol_file:
            protocol_content = protocol_file.read()
            # Use a regular expression to find the number of cycles
            max_cycles = extract_max_cycles(protocol_content)

    for file_name in sorted(os.listdir(temp_dir)):
        if file_name.endswith('.Plateread'):
            cycle = int(file_name.replace('Read', '').replace('.Plateread', ''))

            file_path = os.path.join(temp_dir, file_name)
            with open(file_path, 'rb') as file:
                file.seek(235)  # Skip to the fluorescence data offset

                react_id = 1
                max_wells = 384
                while react_id <= max_wells:
                    bytes_read = file.read(8)
                    if len(bytes_read) < 8:
                        # Not enough data to unpack, likely at the end of the file
                        break

                    fluorescence_value, _ = struct.unpack('ff', bytes_read)
                    well_id = map_react_id_to_well(react_id)

                    # Temporary dictionary to hold the data
                    temp_dict = {
                        'ReactID': react_id,
                        'WellID': well_id,
                        'SampleID': 'sample',
                        'DyeID': dye_id,
                        'Cycle': cycle,
                        'Temperature': temperatures.get(cycle,0),
                        'Fluorescence': fluorescence_value
                    }

                    # Decide whether to append to PCR data or melt curve data
                    if cycle <= max_cycles:
                        data.append(temp_dict)
                    else:
                        melt_data.append(temp_dict)

                    react_id += 1

    # Convert lists to DataFrames
    df = pd.DataFrame(data)
    melt_df = pd.DataFrame(melt_data)
    return df, melt_df

# Function to generate a random string
def generate_random_string(length=8):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))
    
# Function to extract data from XML
def extract_data_from_xml(root):
    data = []
    dye_list = [dye.get('id') for dye in root.findall('.//{http://www.rdml.org}dye')]
    react_count = {}

    for experiment in root.findall('.//{http://www.rdml.org}experiment'):
        for run in experiment.findall('.//{http://www.rdml.org}run'):
            for react in run.findall('.//{http://www.rdml.org}react'):
                react_id = react.get('id')
                react_count[react_id] = react_count.get(react_id, 0) + 1
                order = react_count[react_id] - 1
                dye_id = dye_list[order % len(dye_list)] if order < len(dye_list) else "Unknown"
                well_id = map_react_id_to_well(react_id)
                sample_id = react.find('.//{http://www.rdml.org}sample').get('id')

                for adp in react.findall('.//{http://www.rdml.org}adp'):
                    cycle = adp.find('{http://www.rdml.org}cyc').text
                    temperature = adp.find('{http://www.rdml.org}tmp').text
                    fluor_values = adp.find('{http://www.rdml.org}fluor').text
                    data.append({
                        'ReactID': react_id,
                        'WellID': well_id,
                        'SampleID': sample_id,
                        'DyeID': dye_id,
                        'Cycle': cycle,
                        'Temperature': temperature,
                        'Fluorescence': fluor_values
                    })

    df = pd.DataFrame(data)
    melt_df = pd.DataFrame()
    return df, melt_df   
    
def read_qpcr(uploaded_file):
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    content = uploaded_file.read()

    # Generate a random string of 6 letters and numbers
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    current_working_directory = os.getcwd()
    temp_dir = tempfile.mkdtemp(prefix=f"temp_{random_string}_", dir=current_working_directory)

    if file_extension == '.rdml':

        # Open the zip file
        with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_ref:
            # Extract all the contents into a temporary directory with a random name
            temp_dir = tempfile.mkdtemp(prefix=f"temp_{random_string}_")
            zip_ref.extractall(temp_dir)

            # Assuming there's only one XML file in the zip
            xml_file = [f for f in os.listdir(temp_dir) if f.endswith('.rdml') or f.endswith('.xml')][0]
            xml_file_path = os.path.join(temp_dir, xml_file)

            tree = ET.parse(xml_file_path)

            # Cleanup: Remove the temporary directory after use
            os.remove(xml_file_path)
            os.rmdir(temp_dir)

            return tree.getroot()

    elif file_extension == '.xml':
        # For a regular XML file, use BytesIO to parse it
        tree = ET.ElementTree(ET.fromstring(content))
        return tree.getroot()

    elif file_extension == '.zpcr':
        # Handle zPCR files
        with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_ref:
            # Extract all the contents into a temporary directory with a random name
            zip_ref.extractall(temp_dir)
            # Return the path to the temporary directory containing extracted files
            return temp_dir

    else:
        raise ValueError("Unsupported file format")

def process_pcr_data_file(uploaded_file, pcr_data_basename):
    if uploaded_file is not None:
        with st.spinner("Processing PCR data file..."):
            if uploaded_file.name.endswith('.zpcr'):
                temp_dir = read_qpcr(uploaded_file)
                raw_data, melt_data = extract_data_from_zpcr(temp_dir, pcr_data_basename)
                # Cleanup: Remove the temporary directory and files after use
                for file_name in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file_name))
                os.rmdir(temp_dir)
            else:
                with st.spinner("Processing PCR data file..."):
                    root = read_qpcr(uploaded_file)
                    raw_data, melt_data = extract_data_from_xml(root)
                    melt_data = pd.DataFrame()
            
            # Convert raw_data to DataFrame if it's not already one
            if not isinstance(raw_data, pd.DataFrame):
                raw_data = pd.DataFrame(raw_data)
            
            # Convert DataFrame to TSV format
            tsv_data = raw_data.to_csv(sep='\t', index=False)
            tsv_melt_data = melt_data.to_csv(sep='\t', index=False)
            
            # Create a download button for the TSV data
            st.sidebar.download_button(
                label="Download raw fluorescence data as TSV",
                data=tsv_data,
                file_name=f"{pcr_data_basename}_raw_data.tsv",
                mime="text/tab-separated-values"
            )

            # Create a download button for the TSV data
            st.sidebar.download_button(
                label="Download melt curve fluorescence data as TSV",
                data=tsv_melt_data,
                file_name=f"{pcr_data_basename}_melt_data.tsv",
                mime="text/tab-separated-values"
            )

            return raw_data, melt_data
    return None, None

def map_react_id_to_well(react_id):
    react_id = int(react_id)
    row = chr(65 + (react_id - 1) // 24)
    col = (react_id - 1) % 24 + 1
    return f'{row}{col:02d}'  # Zero-pad the column number

def baseline_subtraction(df):
    df['Cycle'] = pd.to_numeric(df['Cycle'], errors='coerce')
    df['Fluorescence'] = pd.to_numeric(df['Fluorescence'], errors='coerce')
    df['Adjusted Fluorescence'] = df['Fluorescence']

    if not st.session_state['selected_wells']:
        return df, {}

    df = df[df['WellID'].isin(st.session_state['selected_wells'])].copy()
    baseline_end_cycles = {}  # dictionary to store values
    for (well_id, dye_id), group in df.groupby(['WellID', 'DyeID']):
        
        if st.session_state['baseline_cycle_' + dye_id] > 0:
            min_fluorescence = group['Fluorescence'].min()
            group['Adjusted Fluorescence'] = group['Fluorescence'] - min_fluorescence + 1

            # Initialize variables
            baseline_start_cycle = None
            set_point_fluorescence = group['Adjusted Fluorescence'].iloc[st.session_state['baseline_cycle_' + dye_id]]
            last_difference = 0  # Initialize to track the last difference from the set point
            last_increase = 0  # Initialize to track the last increase amount
            exponential_increases_count = 0
            exponential_start_cycle = None

            # Iterate over the cycles, starting from one cycle after the baseline cycle defined in the session state
            for start in range(st.session_state['baseline_cycle_' + dye_id] + 1, len(group)):
                # Calculate the difference from the set point for the current cycle
                current_difference = group['Adjusted Fluorescence'].iloc[start] - set_point_fluorescence

                # Calculate the increase from the last difference
                current_increase = current_difference - last_difference

                # Ensure the current increase is greater than the last increase, indicating accelerating growth
                if current_increase > last_increase and last_increase > 0:
                    exponential_increases_count += 1

                    lookback = st.session_state[f'exponential_cycles_threshold_{dye_id}']
                    # Check if we have a sufficient number of cycles showing accelerating increases
                    if exponential_increases_count == lookback:  # For example, 5 cycles including the set point cycle
                        exponential_start_cycle = group['Cycle'].iloc[start - lookback]  # Adjusting for the look-back
                        break
                else:
                    # Reset the count if the increase is not accelerating
                    exponential_increases_count = 0

                # Update for the next iteration
                last_difference = current_difference
                last_increase = current_increase

            # Identify Baseline Start Cycle if not already set
            if baseline_start_cycle is None:
                baseline_start_cycle = group['Cycle'].iloc[st.session_state['baseline_cycle_' + dye_id]]

            # Identify Baseline Start Cycle if not already set
            if baseline_start_cycle is None:
                baseline_start_cycle = group['Cycle'].iloc[st.session_state['baseline_cycle_' + dye_id]]

            # Baseline end is the cycle before the exponential start
            baseline_end_cycle = exponential_start_cycle - 2 if exponential_start_cycle is not None else None
            
            if dye_id not in baseline_end_cycles:
                baseline_end_cycles[dye_id] = []
                
            if baseline_end_cycle:
                baseline_end_cycles[dye_id].append(baseline_end_cycle)
            
            if baseline_end_cycle is not None:
                baseline_data = group[group['Cycle'].isin([1, baseline_end_cycle])]
                X = baseline_data[['Cycle']]
                y = baseline_data['Adjusted Fluorescence']
                linreg = LinearRegression()
                linreg.fit(X, y)

                # Step 3: Subtract predicted fluorescence from each cycle
                group['Linear Predicted'] = linreg.predict(group[['Cycle']])
                group['Adjusted Fluorescence'] -= group['Linear Predicted']
                   
                # Step 4: Adjust so that the minimum 'Adjusted Fluorescence' is 1
                min_fluorescence = group['Adjusted Fluorescence'].min()
                if min_fluorescence < 1:
                    adjustment_factor = 1 - min_fluorescence
                    group['Adjusted Fluorescence'] += adjustment_factor
                    
                # Step 5: Set adjusted fluorescence value of baseline_end_cycle and before to 1
                group.loc[group['Cycle'] <= baseline_end_cycle, 'Adjusted Fluorescence'] = 1
                                
            # Update the original dataframe with the adjusted values
            df.update(group[['Adjusted Fluorescence']])

        else:
            baseline_end_cycles[dye_id] = None
            
    return df, baseline_end_cycles

def find_steepest_section(df, dye_threshold, baseline_cycle):
    df_filtered = df[(df['Cycle'] > baseline_cycle) & (df['Log Fluorescence'] >= dye_threshold)]
    max_slope = -np.inf
    steepest_section = None

    # Sliding window approach
    for start in range(len(df_filtered) - st.session_state[f'eff_window_size_{dye_id}'] + 1):
        end = start + st.session_state[f'eff_window_size_{dye_id}']
        window = df_filtered.iloc[start:end]
        X = window[['Cycle']]
        y = window['Log Fluorescence']

        linreg = LinearRegression()
        linreg.fit(X, y)
        slope = linreg.coef_[0]

        if slope > max_slope:
            max_slope = slope
            steepest_section = window

    return steepest_section, max_slope

def calculate_average_midpoint_steepest_slope(all_steepest_sections, unique_dyes):
    avg_midpoints = {}
    for dye in unique_dyes:
        midpoints = []
        for (well_id, dye_id), section in all_steepest_sections.items():
            if dye_id == dye:
                midpoint = section['Adjusted Fluorescence'].median()  # Assuming this is how you define the midpoint
                midpoints.append(midpoint)

        if midpoints:
            avg_midpoints[dye] = sum(midpoints) / len(midpoints)
        else:
            avg_midpoints[dye] = None  # or a default value

    return avg_midpoints

def calculate_pcr_efficiency(df, well_id, dye_id):
    # Get the baseline cycle for this dye
    baseline_cycle = st.session_state['baseline_cycle_' + dye_id]

    # Apply log transformation to the 'Adjusted Fluorescence' column
    df['Log Fluorescence'] = np.log(df['Adjusted Fluorescence'])

    # Filter the DataFrame based on WellID and DyeID
    df_filtered = df[(df['WellID'] == well_id) & 
                     (df['DyeID'] == dye_id) & 
                     (df['Cycle'] > baseline_cycle + st.session_state[f'ignore_cycles_{dye_id}'])]

    # Initialize the variables for detecting continuous increase
    continuous_increase_start = None
    continuous_increase_count = 0

    # Iterate through the filtered DataFrame to find the start of continuous increase
    for i in range(1, len(df_filtered)):
        if df_filtered.iloc[i]['Adjusted Fluorescence'] > df_filtered.iloc[i - 1]['Adjusted Fluorescence']:
            continuous_increase_count += 1
            if continuous_increase_count == 1:  # Mark the start of potential continuous increase
                continuous_increase_start = i
            elif continuous_increase_count >= 10:  # Confirm the start if the increase is continuous for at least 10 cycles
                break
        else:
            continuous_increase_count = 0  # Reset if the increase is not continuous

    # Make sure that continuous_increase_start is within bounds
    if continuous_increase_start is not None:
        index = continuous_increase_start + 1
        if not st.session_state['use_steepest_section_for_threshold']:
            log_fluorescence_threshold = st.session_state['log_fluorescence_threshold_' + dye_id]
        if index < len(df_filtered):
            log_fluorescence_threshold = np.log(df_filtered.iloc[index]['Adjusted Fluorescence'])
        else:
            # Handle the case where index is out of bounds
            log_fluorescence_threshold = np.log(df_filtered.iloc[-1]['Adjusted Fluorescence'])
    else:
        return None, None, None, None
        
    if log_fluorescence_threshold < st.session_state['log_fluorescence_threshold_' + dye_id]:
        log_fluorescence_threshold = st.session_state['log_fluorescence_threshold_' + dye_id]

    # Further filter the DataFrame based on the calculated threshold
    df_filtered = df_filtered[df_filtered['Log Fluorescence'] >= log_fluorescence_threshold]

    max_slope = -np.inf  # Initialize to a very small number
    steepest_section = None

    # Sliding window approach to find the steepest section
    for start in range(len(df_filtered) - st.session_state[f'eff_window_size_{dye_id}'] + 1):
        end = start + st.session_state[f'eff_window_size_{dye_id}']
        window = df_filtered.iloc[start:end]
        X = window[['Cycle']]
        y = window['Log Fluorescence']

        linreg = LinearRegression()
        linreg.fit(X, y)
        slope = linreg.coef_[0]

        if slope > max_slope:
            max_slope = slope
            steepest_section = window

    # Calculate PCR efficiency
    efficiency = None
    Cq = None

    if steepest_section is not None and len(steepest_section) > 1:
        first_cycle_fluor = steepest_section.iloc[0]['Adjusted Fluorescence']
        last_cycle_fluor = steepest_section.iloc[-1]['Adjusted Fluorescence']
        cycles = len(steepest_section)
        efficiency = (last_cycle_fluor / first_cycle_fluor) ** (1 / (cycles - 1))
        
        # Fit the linear regression model to the steepest section
        linreg = LinearRegression()
        linreg.fit(steepest_section[['Cycle']], steepest_section['Log Fluorescence'])
        slope = linreg.coef_[0]
        intercept = linreg.intercept_

        # Calculate the x-intercept (Cq)
        if slope != 0:  # To avoid division by zero
            Cq = -intercept / slope + 10  
            # Cq = (st.session_state['log_fluorescence_threshold_' + dye_id] - intercept) / slope
            # print(str(st.session_state['log_fluorescence_threshold_' + dye_id]) + " - " + str(intercept) + " / " + str(slope))
            # print("Cq: " + str(Cq))
        
    return efficiency, steepest_section, max_slope, Cq

def calculate_cq_value(df, threshold=None):
    # Assuming you have a dictionary of threshold factors for each dye
    dye_id = df.iloc[0]['DyeID']
    threshold_factor = 0.5

    # Calculate Cq value for all wells
    above_threshold = df[df['Adjusted Fluorescence'] >= threshold].head(3)
    if above_threshold.empty:
        cq_value = None
    else:
        linreg = LinearRegression()
        X = above_threshold[['Cycle']]
        y = above_threshold['Adjusted Fluorescence']
        linreg.fit(X, y)
        
        # This checks if the regression coefficient is not zero to avoid division by zero
        if linreg.coef_[0] != 0:
            cq_value = (threshold - linreg.intercept_) / linreg.coef_[0]
        else:
            cq_value = None
    
    return cq_value

def extract_numeric_part(sample_name):
    # Finds the last numeric part of the sample name, if present
    return int(re.search(r'\d+$', sample_name).group()) if re.search(r'\d+$', sample_name) else 0

def thousands_formatter(x, pos):
    return '%1.0f' % (x * 1e-3)

def generate_sample_id_to_color(sample_ids, dye_id, standard_color="#9faee5ff", default_color="#1e22aaff"):
    sample_id_to_color = {}
    
    dye_colors = {'FAM': '#1e22aaff', 'HEX': '#78be20ff', 'TEX': '#e4002bff', 'Cy5': '#6d2077ff'}
    other_colors = itertools.cycle(['cyan', 'magenta', 'yellow', 'black', 'orange'])
    
    # Remove 'Standard_' prefixed IDs to get the actual count for color assignment
    non_standard_ids = [sid for sid in sample_ids if 'Standard_' not in sid]
    num_colors_needed = len(non_standard_ids)

    if st.session_state['color_by_samples']:
        # Define a list of preferred color palettes with their maximum color counts
        color_palettes = [
            ("tab10", 10),
            ("Set3", 12),
            ("tab20", 20),
            ("husl", 256)  # 'husl' is a good choice for large distinct color sets
        ]

        # Select the most suitable color palette based on the number of colors needed
        for palette_name, max_colors in color_palettes:
            if num_colors_needed <= max_colors:
                colors = sns.color_palette(palette_name, num_colors_needed)
                break
        else:
            # For 'husl', skip more colors to ensure distinct separation and randomize the placement
            base_colors = sns.color_palette("husl", 256)
            distinct_colors = base_colors[::5]  # Skip every 5 hues to get more distinct colors
            colors = random.sample(distinct_colors, num_colors_needed)  # Randomly sample the required number of colors

        # Shuffle colors to avoid similar colors being placed side by side
        random.shuffle(colors)

        # Assign colors to sample IDs
        color_index = 0
        for sample_id in sample_ids:
            if 'Standard_' in sample_id:
                # Assign distinct colors to each standard sample
                sample_id_to_color[sample_id] = colors[color_index % len(colors)][:3]
                color_index += 1
            else:
                sample_id_to_color[sample_id] = colors[color_index][:3]  # Convert RGBA to RGB
                color_index += 1
    else:
        # Assign default or standard color based on sample ID
        for sample_id in sample_ids:
            if 'Standard_' in sample_id:
                sample_id_to_color[sample_id] = standard_color
            elif dye_id:
                sample_id_to_color[sample_id] = dye_colors[dye_id]
            else:
                sample_id_to_color[sample_id] = default_color

    return sample_id_to_color

def plot_raw_melt_curves_colored(df, dye_id, ax):
    unique_samples = df[df['DyeID'] == dye_id]['SampleID'].unique()
    sample_colors = generate_sample_id_to_color(unique_samples, dye_id)  # Generate colors for each sample

    for sample_id in unique_samples:
        sample_data = df[(df['DyeID'] == dye_id) & (df['SampleID'] == sample_id)]
        for well_id in sample_data['WellID'].unique():
            well_data = sample_data[sample_data['WellID'] == well_id]
            ax.plot(well_data['Temperature'], well_data['Fluorescence'], label=f"Sample {sample_id} Well {well_id}", color=sample_colors[sample_id], linestyle='-', linewidth=0.5)

    ax.set_title(f"Raw Melt Curves for Dye {dye_id}")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Fluorescence")
    
def plot_derivative_melt_curves_with_peaks(df, dye_id, ax, prominence=0.01):
    """Plot the derivative of fluorescence with respect to temperature for melt curves, identify and mark the most prominent peak."""
    unique_samples = df[df['DyeID'] == dye_id]['SampleID'].unique()
    sample_colors = generate_sample_id_to_color(unique_samples, dye_id)  # Generate colors for each sample
    melt_temperatures = []

    for sample_id in unique_samples:
        sample_data = df[(df['DyeID'] == dye_id) & (df['SampleID'] == sample_id)]

        for well_id in sample_data['WellID'].unique():
            well_data = df[(df['DyeID'] == dye_id) & (df['WellID'] == well_id)].sort_values(by='Temperature')
            temperature = well_data['Temperature']
            fluorescence = well_data['Fluorescence']

            # Calculate the negative derivative
            derivative = -np.gradient(fluorescence, temperature)

            # Plot the derivative curve for this well with its corresponding color
            ax.plot(temperature, derivative, label=f"Well {well_id}", color=sample_colors[sample_id], linestyle='-', linewidth=0.5)

            # Find peaks with the highest prominence
            peaks, properties = find_peaks(derivative, prominence=prominence)

            if len(peaks) > 0:
                # Find the most prominent peak
                most_prominent_peak = peaks[np.argmax(properties["prominences"])]
                melt_temperature = temperature.iloc[most_prominent_peak]
                
                melt_temperatures.append({'DyeID': dye_id, 'WellID': well_id, 'MeltTemp': melt_temperature})

                # Mark the most prominent peak on the plot
                ax.scatter(temperature.iloc[most_prominent_peak], derivative[most_prominent_peak], s=20, facecolors='none', edgecolors='red', linewidths=0.5, zorder=5)

    melt_temperatures_df = pd.DataFrame(melt_temperatures)

    ax.set_title(f"Derivative Melt Curves and Peaks for Dye {dye_id}")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("-d(Fluorescence)/dT")
    
    return melt_temperatures_df
    
def plot_melt_curves(df, unique_dyes, prominence=0.01):
    """Plot raw and derivative melt curves side by side for each dye."""
    st.subheader("Melt Curves:")

    for dye_id in unique_dyes:
        # Streamlit columns for side-by-side plots
        
        baseline_cycle = st.session_state['baseline_cycle_' + dye_id]
        col1, col2 = st.columns(2)
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        with col1:
            st.write(f"Raw Melt Curve for {dye_id}")
            # Plot raw melt curves with coloring
            plot_raw_melt_curves_colored(df, dye_id, ax1)

            st.pyplot(fig1)

        with col2:
            st.write(f"Derivative Melt Curve with Peaks for {dye_id}")
            # Plot derivative melt curves with peak identification
            melt_temperatures_df = plot_derivative_melt_curves_with_peaks(df, dye_id, ax2, prominence)
            
            st.pyplot(fig2)
    
    return melt_temperatures_df

def plot_dye_curves(df, cq_thresholds, dye_id, ax, steepest_sections, baseline_end_cycles, log_transform=False,
                    log_fluorescence_threshold=None, baseline_cycle=None, is_last_dye=False):
    if len(df) == 0:
        return

    sample_ids = df['SampleID'].unique()
    sample_id_to_color = generate_sample_id_to_color(sample_ids, dye_id)

    dye_group = df[df['DyeID'] == dye_id]
    grouped_by_well = dye_group.groupby('WellID')

    # Initialize flags for simplified labelling when color_by_samples is False
    if not st.session_state['color_by_samples']:
        standard_label_added = False
        sample_label_added = False

    # Initialize a set to keep track of samples that have been added to the legend
    added_samples = set()

    for well_id, well_group in grouped_by_well:
        well_group_sorted = well_group.sort_values('Cycle')
        sample_id = well_group_sorted['SampleID'].iloc[0]
        color = sample_id_to_color.get(sample_id, "#000000")

        fluorescence_data = np.log(well_group_sorted['Adjusted Fluorescence']) if log_transform else well_group_sorted['Adjusted Fluorescence']

        # Skip label assignment for log-transformed plots
        label = None
        if not log_transform:  # Only assign labels when not log-transformed
            if not st.session_state['color_by_samples']:
                if 'Standard_' in sample_id:
                    if not standard_label_added:
                        label = 'Standard'
                        standard_label_added = True
                else:
                    if not sample_label_added:
                        label = 'Sample'
                        sample_label_added = True
            else:
                # For non-log-transformed plots with color by samples enabled,
                # add the sample ID as the label if it hasn't been added to the legend yet.
                if sample_id not in added_samples:
                    label = sample_id
                    added_samples.add(sample_id)  # Mark this sample as added to avoid duplicate labels

        ax.plot(well_group_sorted['Cycle'], fluorescence_data, linestyle='-', color=color, linewidth=0.5, label=label)

        # Plot steepest section without adding to legend
        if (well_id, dye_id) in steepest_sections and st.session_state["show_steepest_section"]:
            steepest_section = steepest_sections[(well_id, dye_id)]
            fluorescence_steepest = np.log(steepest_section['Adjusted Fluorescence']) if log_transform else steepest_section['Adjusted Fluorescence']
            ax.scatter(steepest_section['Cycle'], fluorescence_steepest, s=20, facecolors='none', edgecolors='red', linewidths=0.5, zorder=5)

    # Add legend if simplified labelling was used, or if color_by_samples is True and there are labels to show
    if not st.session_state['color_by_samples'] and (standard_label_added or sample_label_added) or st.session_state['color_by_samples']:
        ax.legend()

    if cq_thresholds and cq_thresholds.get(dye_id):
        cq_threshold = np.log(cq_thresholds[dye_id]) if log_transform else cq_thresholds[dye_id]
        ax.axhline(y=cq_threshold, color='blue', linestyle='--', linewidth=1, label='Cq threshold' if log_transform else None)

    # if log_fluorescence_threshold is not None:
        # ax.axhline(y=log_fluorescence_threshold, color='black', linestyle='--', linewidth=1, label='Log Fluorescence Threshold')

    if baseline_cycle:
        ax.axvline(x=baseline_cycle, color='green', linestyle='--', linewidth=1, label='Baseline Cycle' if log_transform else None)
        if st.session_state[f'ignore_cycles_{dye_id}'] > 0:
            ignore_cycles = st.session_state[f'ignore_cycles_{dye_id}']
            ignore_until_cycle = baseline_cycle + st.session_state[f'ignore_cycles_{dye_id}']
            ax.axvline(x=ignore_until_cycle, color='red', linestyle='--', linewidth=1, label=f'Ignore After {ignore_cycles} Cycles' if log_transform else None)

    ax.set_title(f'{"Log-Transformed " if log_transform else ""}Amplification Curve for {dye_id}')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Log Adjusted Fluorescence' if log_transform else 'Adjusted Fluorescence')

    if not log_transform:
        def thousands_formatter(x, pos):
            return '{:02d}'.format(int(x * 1e-2))

        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.set_ylabel('Adjusted Fluorescence (x100)')
    else:
        def zero_padding_formatter(x, pos):
            # Assuming you want at least 4 digits with leading zeros
            return '{:02d}'.format(int(x))

        # Apply the formatter to the current y-axis
        ax.yaxis.set_major_formatter(FuncFormatter(zero_padding_formatter))
        ax.set_ylim(bottom=st.session_state['log_fluorescence_threshold_' + dye_id])

    # Sort the legend labels as required
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True,
              ncol=2 if log_transform else 4)

def sort_wells(well_list):
    def well_sort_key(well):
        # Split the well identifier into alphabetical and numerical parts
        match = re.match(r"([A-Za-z]+)([0-9]+)", well)
        if match:
            # Convert the numerical part to an integer for correct sorting
            return match.group(1), int(match.group(2))
        return well, 0

    return sorted(well_list, key=well_sort_key)

def handle_row_button(row, available_wells):
    # Wells in the row based on available wells in the dataset
    row_wells = sorted([well for well in available_wells if well.startswith(row)])

    # Current selected wells, ensuring they are also zero-padded
    current_selected_padded = set(f"{well[0]}{int(well[1:]):02d}" for well in st.session_state['selected_wells'])

    if current_selected_padded.issuperset(set(row_wells)):
        # Remove all wells in the row from the selection
        st.session_state['selected_wells'] = sorted(list(current_selected_padded - set(row_wells)))
    else:
        # Add all wells in the row to the selection
        st.session_state['selected_wells'] = sorted(list(current_selected_padded.union(set(row_wells))))
    
    # Rerun the script to update the app with the new state
    st.rerun()

def handle_column_button(column, available_wells):
    # Convert column number to string with leading zeros for consistency
    column_str = f"{int(column):02d}"

    # Wells in the column based on available wells in the dataset
    col_wells = sorted([well for well in available_wells if well.endswith(column_str)])

    # Remove zero-padding from col_wells to match selected_wells
    col_wells_no_padding = [well[:-1] + str(int(well[-1])) if well.endswith("0") else well for well in col_wells]

    current_selected = set(st.session_state['selected_wells'])

    if current_selected.issuperset(set(col_wells_no_padding)):
        # Remove all wells in the column from the selection
        st.session_state['selected_wells'] = sorted(list(current_selected - set(col_wells_no_padding)))
    else:
        # Add all wells in the column to the selection
        st.session_state['selected_wells'] = sorted(list(current_selected.union(set(col_wells_no_padding))))

    # Rerun the script to update the app with the new state
    st.rerun()

def handle_sample_button(sample_id, current_data, available_wells):
    # Get the wells associated with the selected SampleID
    sample_wells = current_data[current_data['SampleID'] == sample_id]['WellID'].unique()

    # Current selected wells, ensuring they are also zero-padded
    current_selected_padded = set(f"{well[0]}{int(well[1:]):02d}" for well in st.session_state['selected_wells'])

    if current_selected_padded.issuperset(set(sample_wells)):
        # Remove all wells associated with the SampleID from the selection
        st.session_state['selected_wells'] = sorted(list(current_selected_padded - set(sample_wells)))
    else:
        # Add all wells associated with the SampleID to the selection
        st.session_state['selected_wells'] = sorted(list(current_selected_padded.union(set(sample_wells))))

    # Rerun the script to update the app with the new state
    st.rerun()

def calculate_buttons_per_row(sample_ids):
    # Calculate the maximum number of buttons per row (4) and the number of rows
    max_buttons_per_row = 5
    total_buttons = len(sample_ids)
    rows = total_buttons // max_buttons_per_row
    remainder = total_buttons % max_buttons_per_row

    # Determine the number of buttons in the last row
    buttons_per_row = [max_buttons_per_row] * rows
    if remainder > 0:
        buttons_per_row.append(remainder)

    return buttons_per_row

def display_sample_buttons(sample_ids, current_data, available_wells):
    buttons_per_row = calculate_buttons_per_row(sample_ids)

    i = 0
    for row_buttons in buttons_per_row:
        cols = st.columns(row_buttons)
        for _ in range(row_buttons):
            sample_id = sample_ids[i]
            with cols[_]:
                if st.button(sample_id, on_click=lambda: st.session_state.update({'plot_and_calculate': False})):
                    handle_sample_button(sample_id, current_data, available_wells)
            i += 1

def generate_well_table(available_wells, sample_id_to_color, well_id_to_sample_id, well_id_to_sampletype):
    top_rows = [chr(i) for i in range(65, 73)]  # A-H
    bottom_rows = [chr(i) for i in range(73, 81)]  # I-P
    left_columns = list(range(1, 13))  # 1-12
    right_columns = list(range(13, 25))  # 13-24

    # Define CSS styles
    css_style = """
    <style>
        table, th, td {
            font-family: Arial, sans-serif;
            font-size: 10px;
            text-align: center;
            border: 1px solid black;
        }
        .circle {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            text-align: center;
        }
        .white-square-table {
            width: 10px;
            height: 10px;
            background-color: white;
            display: inline-block;
            margin: auto;
        }
        .white-circle-table {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: white;
            border: 1px solid black;
            text-align: center;
        }
        .legend-white-square {
            width: 10px;
            height: 10px;
            background-color: white;
            display: inline-block;
            margin: auto;
            border: 1px solid black;
        }
        .legend-row {
            display: flex;
            margin-top: 5px;
        }
        .legend-cell {
            margin-right: 10px;
            display: flex;
            align-items: center;
        .legend-text {
            font-size: 10px; 
        }
        }
    </style>
    """

    def create_quadrant_table(rows, columns, is_left_table=True):
        margin_right = "35px" if is_left_table else "0px"
        table_html = css_style
        table_html += f"<table style='border-collapse: collapse; display: inline-block; margin: 0px; margin-right: {margin_right}; border: 1.5px solid black;'>"
        table_html += "<tr><th style='width: 10px; height: 10px;'></th>"

        for col in columns:
            col_label = f"{int(col):02d}" if col < 10 else str(col)
            table_html += f"<th style='width: 10px; height: 10px;'>{col_label}</th>"
        table_html += "</tr>"

        for row in rows:
            table_html += f"<tr><td style='width: 10px; height: 10px;'>{row}</td>"
            for col in columns:
                well_id = f"{row}{col:02d}"
                sample_id = well_id_to_sample_id.get(well_id)
                # Check if the well is selected, available, or not available
                if well_id in selected_wells_set:
                    # Highlight the well with the sample's color if it's selected
                    color_hex = matplotlib.colors.rgb2hex(sample_id_to_color.get(sample_id, "#FFFFFF"))  # Default to white if missing
                    label = f"<div class='circle' style='background-color: {color_hex};'></div>"
                elif well_id in available_wells:
                    # Show the well as available but not selected
                    label = "<div class='white-circle-table'></div>"
                else:
                    # Show the well as not available
                    label = "<div class='white-square-table'></div>"
                table_html += f"<td style='width: 10px; height: 10px;'>{label}</td>"
            table_html += "</tr>"
        table_html += "</table>"
        return table_html

    def sort_samples_by_well(sample_items, well_id_to_sample_id):
        """Sorts sample items based on the first appearance of their well ID in the plate."""
        sample_to_first_well = {sample_id: min([well for well, sample in well_id_to_sample_id.items() if sample == sample_id], key=lambda x: (x[0], int(x[1:]))) for sample_id, _ in sample_items}
        return sorted(sample_items, key=lambda item: (sample_to_first_well[item[0]][0], int(sample_to_first_well[item[0]][1:])))

    def generate_legend(samples, include_well_availability=False):
        # Sort the sample items before generating legend entries based on well order
        sample_items = sort_samples_by_well(list(samples.items()), well_id_to_sample_id)

        # Start the legend HTML without the 'Legend:' label
        legend_html = "<div style='margin-left: 5px;'>"

        if include_well_availability:
            # Combine the "Available Well" and "Unavailable Well" in a single row for side-by-side display
            legend_html += "<div class='legend-row'>" \
                           "<div class='legend-cell'><div class='white-circle-table'></div> <span class='legend-text'>Available Well</span></div>" \
                           "<div class='legend-cell'><div class='legend-white-square'></div> <span class='legend-text'>Unavailable Well</span></div>" \
                           "</div>"

        for i in range(0, len(sample_items), 3):
            legend_html += "<div class='legend-row'>"
            for sample_id, color in sample_items[i:i+3]:
                color_hex = matplotlib.colors.rgb2hex(color)
                legend_html += f"<div class='legend-cell'><div class='circle' style='background-color: {color_hex};'></div> <span class='legend-text'>{sample_id}</span></div>"
            legend_html += "</div>"

        legend_html += "</div>"
        return legend_html
    
    # Filter for selected wells and sample IDs
    selected_wells_set = set(st.session_state['selected_wells'])
    selected_wells = st.session_state.get('selected_wells', [])
    selected_well_id_to_sample_id = {well_id: sample_id for well_id, sample_id in well_id_to_sample_id.items() if well_id in selected_wells}

    # Resolve sample types for selected sample IDs using well IDs
    selected_well_id_to_sample_type = {well_id: well_id_to_sampletype[well_id] for well_id in selected_well_id_to_sample_id if well_id in well_id_to_sampletype}

    # Filtered sample IDs based on selected wells
    selected_sample_ids = set(selected_well_id_to_sample_id.values())

    # Filtered sample ID to color mapping based on selected sample IDs
    filtered_sample_id_to_color = {sample_id: color for sample_id, color in sample_id_to_color.items() if sample_id in selected_sample_ids}

    # Sort the samples for the legends based on well order
    sorted_samples = sort_samples_by_well(list(filtered_sample_id_to_color.items()), well_id_to_sample_id)

    # Split the sorted samples for the top and bottom legends
    top_samples = dict(sorted_samples[:21])
    bottom_samples = dict(sorted_samples[21:])

    top_legend_html = generate_legend(top_samples, include_well_availability=True)
    bottom_legend_html = generate_legend(bottom_samples)

    # Create tables for each quadrant
    top_table_html = create_quadrant_table(top_rows, left_columns, is_left_table=True) + \
                     create_quadrant_table(top_rows, right_columns, is_left_table=False)
    bottom_table_html = create_quadrant_table(bottom_rows, left_columns, is_left_table=True) + \
                        create_quadrant_table(bottom_rows, right_columns, is_left_table=False)

    top_combined_html = f"{css_style}<div style='display: flex; align-items: flex-start;'>{top_table_html}{top_legend_html}</div>"
    bottom_combined_html = f"{css_style}<div style='display: flex; align-items: flex-start;'>{bottom_table_html}{bottom_legend_html}</div>"

    st.components.v1.html(top_combined_html)
    st.components.v1.html(bottom_combined_html)
    
def plot_standard_curve(ax, dye_standards_data, dye, regression_line=None, standard_color="#9faee5ff", default_color="#1e22aaff", ):
    # Apply log transformation to the standard concentration
    log_std_conc_pm = np.log10(dye_standards_data['std_conc_pm'])

    # Plot actual data points with log-transformed x-axis
    ax.scatter(log_std_conc_pm, dye_standards_data['Cq'], color=standard_color)

    # Set x-axis ticks to align with the log-transformed standard concentration values
    ax.set_xticks(log_std_conc_pm)

    # Label x-axis ticks with the corresponding non-log-transformed standard concentration values
    # Format labels to show significant figures or scientific notation as preferred
    ax.set_xticklabels([f'{value:.1e}' for value in dye_standards_data['std_conc_pm']])

    slope = None
    intercept = None

    # If regression line data is provided, plot it
    if regression_line:
        x_values, y_values, linreg = regression_line
        ax.plot(x_values, y_values, color=default_color, label=f'Regression Line\ny = {linreg.coef_[0]:.4f}x + {linreg.intercept_:.3f}')

        slope = linreg.coef_[0]
        intercept = linreg.intercept_
        
        efficiency = (10 ** (-1 / slope) - 1) * 100

        # Calculate R-squared (R²) value
        residuals = dye_standards_data['Cq'] - linreg.predict(log_std_conc_pm.values.reshape(-1, 1))
        ss_residual = np.sum(residuals**2)
        ss_total = np.sum((dye_standards_data['Cq'] - np.mean(dye_standards_data['Cq']))**2)
        r_squared = 1 - (ss_residual / ss_total)

        # Annotate plot with R-squared value and linear equation
        annotation_text = f'$R^2$ = {r_squared:.4f}\n'
        annotation_text += f'y = {linreg.coef_[0]:.4f}x + {linreg.intercept_:.2f}\n'
        annotation_text += f'efficiency = {efficiency:.1f}%\n'
        ax.annotate(annotation_text, xy=(0.67, 0.8), xycoords='axes fraction', fontsize=10, color='black')

    # Customize the plot
    ax.set_xlabel('Standard pM Concentration (log scale)')
    ax.set_ylabel('Cq Value')
    ax.set_title(f'Standard Curve for {dye}')

    return slope, intercept

def perform_linear_regression(dye_avg_cq_data):
    # Apply log transformation to the standard concentration
    log_std_conc_pm = np.log10(dye_avg_cq_data['std_conc_pm'])

    # Perform linear regression using the log-transformed concentrations
    X = log_std_conc_pm.values.reshape(-1, 1)  # Convert to 2D array for sklearn
    y = dye_avg_cq_data['Cq']
    
    # Create a mask for rows where y is not NaN
    mask = ~np.isnan(y)

    # Apply the mask to X and y to filter out rows with NaN in y
    X_filtered = X[mask]
    y_filtered = y[mask]

    linreg = LinearRegression()
    linreg.fit(X_filtered, y_filtered)
    

    r_squared = r2_score(y_filtered, linreg.predict(X_filtered))

    return linreg, r_squared, log_std_conc_pm  
   
def upload_files():
    uploaded_PCR_file = st.sidebar.file_uploader("Upload your PCR data file", type=['rdml', 'xml', 'zpcr'])
    
    file_path = "CyclusK_qPCR_labelling_template_384_well_including_KAPA_standards.tsv"
    lines = read_tsv_lines(file_path)
    download_tsv(lines)
    
    uploaded_labelling_file = st.sidebar.file_uploader("Upload your labelling file", type=['tsv', 'txt'])
    pcr_data_basename = os.path.splitext(os.path.basename(uploaded_PCR_file.name))[0] if uploaded_PCR_file else 'qPCR'
     
    return uploaded_PCR_file, uploaded_labelling_file, pcr_data_basename

def process_labelling_file(uploaded_file_tsv):
    if uploaded_file_tsv is not None:
        with st.spinner("Processing labelling file..."):
            labelling_data = pd.read_csv(uploaded_file_tsv, sep='\t')
            if 'well' in labelling_data.columns:
                labelling_data['well'] = labelling_data['well'].apply(lambda x: f"{x[0]}{int(x[1:]):02d}")
            else:
                st.error("labelling data must contain 'well' column.")
        return labelling_data
    return None

def merge_and_preprocess_data(raw_data, melt_data, labelling_data, pcr_data_basename):
    if raw_data is not None and 'WellID' in raw_data.columns and labelling_data is not None:
        raw_data = pd.merge(raw_data, labelling_data, left_on=["WellID", "DyeID"], right_on=["well", "filter"], how="inner")
        raw_data['SampleID'] = raw_data['sample']
        raw_data.drop(columns=['sample'], inplace=True)
        
        # Convert DataFrame to TSV format
        tsv_data = raw_data.to_csv(sep='\t', index=False)
        
        # Create a download button for the TSV data
        st.sidebar.download_button(
            label="Download labelled fluorescence data as TSV",
            data=tsv_data,
            file_name=f"{pcr_data_basename}_labelled_data.tsv",
            mime="text/tab-separated-values"
        )                
    return raw_data, melt_data

def setup_analysis_parameters(subtracted_data):
    unique_dyes = subtracted_data['DyeID'].unique()

    # Generic (non-dye-specific) checkboxes
    st.session_state['use_steepest_section_for_threshold'] = st.sidebar.checkbox('Use individual regression for Cq', value=True)
    st.session_state["show_steepest_section"] = st.sidebar.checkbox('Plot Steepest Section', value=True)
    st.session_state['color_by_samples'] = st.sidebar.checkbox('Colour by sample', value=False)
    st.session_state['kapa_analysis'] = st.sidebar.checkbox('Include KAPA analysis', value=True)

    # Initialize dictionaries for dye-specific settings
    ignore_cycles = {}
    eff_window_size = {}
    exponential_cycles_thresholds = {}
    baseline_cycles = {}
    log_fluorescence_thresholds = {}

    for dye in unique_dyes:
        # Title for dye-specific sliders
        st.sidebar.markdown(f"### {dye} Sliders")
        
        # Dye-specific sliders
        log_fluorescence_thresholds[dye] = st.sidebar.slider(f"Log Adjusted Fluorescence threshold ({dye})", 0.0, 10.0, 5.0, key=f'log_fluorescence_threshold_{dye}')
        eff_window_size[dye] = st.sidebar.slider(f"Steepest section cycle window ({dye})", 3, 10, 4, key=f'eff_window_size_{dye}')
        ignore_cycles[dye] = st.sidebar.slider(f"Start cycle for steepest section ({dye})", 0, 10, 0, key=f'ignore_cycles_{dye}')
        exponential_cycles_thresholds[dye] = st.sidebar.slider(f"Exponential Cycles Threshold ({dye})", 3, 10, 4, key=f'exponential_cycles_threshold_{dye}')
        baseline_cycles[dye] = st.sidebar.slider(f"Minimum baseline cycle ({dye})", 0, 10, 1, key=f'baseline_cycle_{dye}')

    return unique_dyes

def analyze_data(subtracted_data, unique_dyes, labelling_data):
    all_steepest_sections = {}
    all_efficiencies = {}
    results = []
    
    cq_thresholds = {}
    cq_values = {}
    efficiencies = {}
    steepest_sections = {}
    max_slopes = {}
    
    for (sample_id, well_id, dye_id), group_data in subtracted_data.groupby(['SampleID', 'WellID', 'DyeID']):
        efficiency, steepest_section, max_slope, Cq = calculate_pcr_efficiency(group_data, well_id, dye_id)
        if steepest_section is not None:
            cq_threshold = steepest_section['Adjusted Fluorescence'].mean()
        else:
            cq_threshold = 1
            
        if dye_id in efficiencies:
            efficiencies[dye_id][well_id] = efficiency
            cq_thresholds[dye_id][well_id] = cq_threshold
            cq_values[dye_id][well_id] = Cq
            steepest_sections[dye_id][well_id] = steepest_section
            max_slopes[dye_id][well_id] = max_slopes
        else:
            efficiencies[dye_id] = {well_id:efficiency}
            cq_thresholds[dye_id] = {well_id:cq_threshold}
            cq_values[dye_id] = {well_id:Cq}
            steepest_sections[dye_id] = {well_id:steepest_section}
            max_slopes[dye_id] = {well_id:max_slope}
            
    mean_cq_thresholds = {}     
    for (sample_id, well_id, dye_id), group_data in subtracted_data.groupby(['SampleID', 'WellID', 'DyeID']):
        efficiency = efficiencies[dye_id][well_id]
        steepest_section = steepest_sections[dye_id][well_id]
        max_slope = max_slopes[dye_id][well_id]

        # Calculate the fluorescence threshold based on the steepest section if the checkbox is checked
        if st.session_state['use_steepest_section_for_threshold'] and steepest_section is not None and not steepest_section.empty:
            cq_threshold = cq_thresholds[dye_id][well_id]
            cq_value = cq_values[dye_id][well_id]
            mean_cq_thresholds[dye_id] = None
        else:
            cq_threshold = np.mean(list(cq_thresholds[dye_id].values()))
            cq_value = calculate_cq_value(group_data, cq_threshold)
            if dye_id not in mean_cq_thresholds:
                mean_cq_thresholds[dye_id] = cq_threshold      

        result = {
            'SampleID': sample_id,
            'WellID': well_id,
            'DyeID': dye_id,
            'Cq': "{:.2f}".format(cq_value) if cq_value is not None else 'N/A',
            'Efficiency': "{:.2f}".format(efficiency) if efficiency is not None else 'N/A',
            '% Efficiency': "{:.0f}%".format(efficiency / 2 * 100) if efficiency else 'N/A',
        }

        if labelling_data is not None:
            result.update({
                'sampletype': group_data.iloc[0]['sampletype'],
                'std_conc_pm': group_data.iloc[0]['std_conc_pm'],
                'dilutionfactor': group_data.iloc[0]['dilutionfactor'],
                'libraryfragmentsize': group_data.iloc[0]['libraryfragmentsize'],
            })

        results.append(result)
        if steepest_section is not None and not steepest_section.empty:
            all_steepest_sections[(well_id, dye_id)] = steepest_section

    results_df = pd.DataFrame(results)
    return results_df, all_steepest_sections, mean_cq_thresholds

def plot_amplification_curves(subtracted_data, cq_thresholds, unique_dyes, all_steepest_sections, baseline_end_cycles):
    st.subheader("Amplification_Curves:")
    for dye_id in unique_dyes:
        baseline_cycle = st.session_state['baseline_cycle_' + dye_id]
        col1, col2 = st.columns(2)
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        filtered_data = subtracted_data[(subtracted_data['DyeID'] == dye_id) & subtracted_data['WellID'].isin(st.session_state['selected_wells'])]

        with col1:
            st.write(f"Raw Amplification Curve for {dye_id}")
            plot_dye_curves(df=filtered_data, 
                            cq_thresholds=cq_thresholds, 
                            dye_id=dye_id, 
                            ax=ax1, 
                            steepest_sections=all_steepest_sections, 
                            baseline_end_cycles=baseline_end_cycles, 
                            log_transform=False, 
                            log_fluorescence_threshold=None, 
                            baseline_cycle=baseline_cycle, 
                            is_last_dye=(dye_id == unique_dyes[-1]),)
            st.pyplot(fig1)

        with col2:
            st.write(f"Log-Transformed Amplification Curve for {dye_id}")
            plot_dye_curves(df=filtered_data, 
                            cq_thresholds=cq_thresholds, 
                            dye_id=dye_id, 
                            ax=ax2, 
                            steepest_sections=all_steepest_sections, 
                            baseline_end_cycles=baseline_end_cycles, 
                            log_transform=True, 
                            log_fluorescence_threshold=st.session_state['log_fluorescence_threshold_' + dye_id], 
                            baseline_cycle=baseline_cycle, 
                            is_last_dye=(dye_id == unique_dyes[-1]),)
            st.pyplot(fig2)

def display_results_table(results_df, labelling_data, pcr_data_basename):
    st.subheader("Results Table:")

    if not st.session_state['selected_wells']:
        st.warning("Please select wells to display results.")
        return

    def well_sort_key(well):
        match = re.match(r"([A-Za-z]+)([0-9]+)", well)
        return (match.group(1), int(match.group(2)))

    filtered_results_df = results_df[results_df['WellID'].isin(st.session_state['selected_wells'])].copy()
    filtered_results_df['WellID'] = pd.Categorical(filtered_results_df['WellID'], categories=sorted(filtered_results_df['WellID'].unique(), key=well_sort_key), ordered=True)
    filtered_results_df.sort_values(by=['DyeID', 'WellID'], inplace=True)
    
    columns = ['SampleID', 'WellID', 'DyeID', 'Cq', '% Efficiency']

    if 'MeltTemp' in filtered_results_df.columns: 
        columns.append('MeltTemp')
    
    if labelling_data is not None:
        columns.extend(['sampletype', 'std_conc_pm', 'dilutionfactor', 'libraryfragmentsize'])

    st.dataframe(filtered_results_df[columns])
       
    st.download_button(
        label="Download results as TSV",
        data=filtered_results_df.to_csv(sep='\t', index=False),
        file_name=f'{pcr_data_basename}_pcr_analysis_results.tsv',
        mime='text/tsv'
    )

def generate_standard_curve_summary(df, pcr_data_basename, dye_id):
    # Filter for only standard samples
    standard_curve_df = df[df['SampleID'].str.contains("Standard_")].copy()

    # Convert 'Cq' to numeric, handling non-numeric values
    standard_curve_df['Cq'] = pd.to_numeric(standard_curve_df['Cq'], errors='coerce')

    # Calculate the average Cq for each standard
    standard_averages = standard_curve_df.groupby('SampleID')['Cq'].mean().reset_index(name='Average Cq')

    # Sort by standard number to ensure correct Delta Cq calculation
    standard_averages['Std Number'] = standard_averages['SampleID'].str.extract(r'(\d+)').astype(int)
    standard_averages.sort_values('Std Number', inplace=True)

    # Calculate Delta Cq as the difference between successive standards
    standard_averages['Delta Cq'] = standard_averages['Average Cq'].diff().fillna(0) * -1  # Multiplying by -1 to invert the difference order

    # Round the 'Average Cq' and 'Delta Cq' to 2 decimal places
    standard_averages['Average Cq'] = standard_averages['Average Cq'].round(2)
    standard_averages['Delta Cq'] = standard_averages['Delta Cq'].round(2)

    # Merge the average and delta Cq values back to the original standard dataframe
    standard_curve_summary_df = standard_curve_df[['SampleID', 'std_conc_pm']].drop_duplicates().merge(
        standard_averages.drop(columns=['Std Number']), on='SampleID')

    # Rename columns for clarity
    standard_curve_summary_df.rename(columns={
        'SampleID': 'Std #',
        'std_conc_pm': 'Conc (pM)'
    }, inplace=True)

    # Sort by standard number for final presentation
    standard_curve_summary_df['Std Number'] = standard_curve_summary_df['Std #'].str.extract(r'(\d+)').astype(int)
    standard_curve_summary_df.sort_values('Std Number', inplace=True)
    standard_curve_summary_df.reset_index(drop=True, inplace=True)
    standard_curve_summary_df.drop(columns=['Std Number'], inplace=True)
    
    # Display the dataframe
    st.subheader("Review Cq values for DNA Standards:")
    st.dataframe(standard_curve_summary_df)
    
    st.download_button(
        label="Download results as TSV",
        data=standard_curve_summary_df.to_csv(sep='\t', index=False),
        file_name=f'{pcr_data_basename}_{dye_id}_standard_curve_table.tsv',
        mime='text/tsv',
        key=f'{pcr_data_basename}_{dye_id}_standard_curve_table'
    )

def calculate_qpcr_results(results_df, slope, intercept, pcr_data_basename, dye_id):
    FRAGMENT_MOLAR_MASS_FACTOR = 617.9  # g/mol
    
    # Convert 'SampleID' to string and exclude standard samples
    results_df['SampleID'] = results_df['SampleID'].astype(str)
    non_standard_df = results_df[~results_df['SampleID'].str.contains("Standard_", case=False, na=False)].copy()
    if 'sampletype' in non_standard_df:
        non_standard_df = non_standard_df[~non_standard_df['sampletype'].str.contains("standard", case=False, na=False)].copy()
    
    # Group by 'SampleID' and 'Dilution' for subsequent calculations
    grouped = non_standard_df.groupby(['SampleID', 'dilutionfactor'])

    # Use .loc to avoid SettingWithCopyWarning when setting new column values after groupby
    non_standard_df.loc[:, 'Average Cq'] = grouped['Cq'].transform('mean')
    non_standard_df.loc[:, 'Difference'] = non_standard_df['Cq'] - non_standard_df['Average Cq']
    non_standard_df.loc[:, 'log_concentration_pM'] = (non_standard_df['Average Cq'] - intercept) / slope
    non_standard_df.loc[:, 'Average concentration (pM)'] = 10 ** non_standard_df['log_concentration_pM']
    non_standard_df.loc[:, 'Size-adjusted concentration (pM)'] = non_standard_df['Average concentration (pM)'] * (452 / non_standard_df['libraryfragmentsize'])
    non_standard_df.loc[:, 'Concentration of undiluted library (pM)'] = non_standard_df['Size-adjusted concentration (pM)'] * non_standard_df['dilutionfactor']
    non_standard_df.loc[:, 'Concentration of undiluted library (nM)'] = non_standard_df['Concentration of undiluted library (pM)'] / 1000
    non_standard_df.loc[:, 'Concentration of undiluted library (ng/µL)'] = (
        (non_standard_df['Concentration of undiluted library (nM)'] / 10**9) *
        (non_standard_df['libraryfragmentsize'] * FRAGMENT_MOLAR_MASS_FACTOR) *
        10**9 / 10**6
    )

    # Sort by 'SampleID' and 'dilutionfactor' in ascending order
    non_standard_df.sort_values(by=['SampleID', 'dilutionfactor'], ascending=True, inplace=True)

    # Group by 'SampleID' only for Delta Cq calculation
    grouped_sample = non_standard_df.groupby('SampleID')
    
    non_standard_df['Working concentration (pM)'] = grouped_sample['Concentration of undiluted library (pM)'].transform('median')
    non_standard_df['Working concentration (nM)'] = grouped_sample['Concentration of undiluted library (nM)'].transform('median')
    non_standard_df['Working concentration (ng/µL)'] = grouped_sample['Concentration of undiluted library (ng/µL)'].transform('median')
    non_standard_df.loc[:, 'Delta Cq'] = grouped_sample['Average Cq'].transform(lambda x: x.diff())
    
    # Identify the first row in each SampleID group
    first_row_mask = grouped_sample.cumcount() == 0

    # Apply the mask to 'Working concentration' columns, setting values to None for rows that are not the first in their group
    for col in ['Working concentration (pM)', 'Working concentration (nM)', 'Working concentration (ng/µL)']:
        non_standard_df.loc[~first_row_mask, col] = None

    # Group by 'SampleID' only for % Deviation calculation
    lowest_concentration = grouped_sample['Concentration of undiluted library (nM)'].transform('first')
    non_standard_df.loc[:, '% Deviation'] = ((lowest_concentration - non_standard_df['Concentration of undiluted library (nM)']) / lowest_concentration) * 100

    # Round values as per specified precision
    cols_to_round = {
        'Cq': 1,
        'Difference': 1,
        'Average Cq': 1,
        'Delta Cq': 1,
        'log_concentration_pM': 2,
        'Average concentration (pM)': 3,
        'Size-adjusted concentration (pM)': 3,
        'Concentration of undiluted library (pM)': 0,
        'Concentration of undiluted library (nM)': 2,
        '% Deviation': 1,
        'Concentration of undiluted library (ng/µL)': 1,
        'Working concentration (pM)' : 0,
        'Working concentration (nM)' : 1,
        'Working concentration (ng/µL)' : 1
    }

    for col, decimals in cols_to_round.items():
        non_standard_df.loc[:, col] = non_standard_df[col].round(decimals)

    # Create a mask to identify the first row of each group
    first_row_mask = non_standard_df.groupby('SampleID').cumcount() == 0

    # Set the values to None for rows that are not the first row in each group
    for col in ['Working concentration (pM)', 'Working concentration (nM)', 'Working concentration (ng/µL)']:
        non_standard_df.loc[~first_row_mask, col] = None
        
    # Create a mask to identify the first row of each group based on 'SampleID' and 'dilutionfactor'
    first_row_mask = non_standard_df.groupby(['SampleID', 'dilutionfactor']).cumcount() == 0

    # Set the values to None for rows that are not the first row in each group, except 'Working concentration' columns
    columns_to_mask = ['Average Cq', 'Delta Cq', 'log_concentration_pM', 'Average concentration (pM)',
                       'Size-adjusted concentration (pM)', 'Concentration of undiluted library (pM)',
                       'Concentration of undiluted library (nM)', '% Deviation', 'Concentration of undiluted library (ng/µL)']
    for col in columns_to_mask:
        non_standard_df.loc[~first_row_mask, col] = None

    # Prepare the final DataFrame with the specified columns
    output_columns = [
        'SampleID', 'dilutionfactor', 'libraryfragmentsize', 'Cq',
        'Difference', 'Average Cq', 'Delta Cq', 'log_concentration_pM',
        'Average concentration (pM)', 'Size-adjusted concentration (pM)',
        'Concentration of undiluted library (pM)', 'Concentration of undiluted library (nM)',
        '% Deviation', 'Concentration of undiluted library (ng/µL)', 'Working concentration (pM)',
        'Working concentration (nM)', 'Working concentration (ng/µL)'
    ]

    # Create the output DataFrame
    output_df = non_standard_df[output_columns].copy()

    # Map the calculated columns to the corresponding output column names
    output_df = pd.DataFrame({
        'SampleID': non_standard_df['SampleID'],
        'Dilution': non_standard_df['dilutionfactor'],
        'Average fragment length (bp)': non_standard_df['libraryfragmentsize'],
        'Cq': non_standard_df['Cq'],
        'Difference': non_standard_df['Difference'],
        'Average Cq': non_standard_df['Average Cq'],
        'Delta Cq': non_standard_df['Delta Cq'],
        'log (concentration pM)': non_standard_df['log_concentration_pM'],
        'Average concentration (pM)': non_standard_df['Average concentration (pM)'],
        'Size-adjusted concentration (pM)': non_standard_df['Size-adjusted concentration (pM)'],
        'Concentration of undiluted library (pM)': non_standard_df['Concentration of undiluted library (pM)'],
        'Concentration of undiluted library (nM)': non_standard_df['Concentration of undiluted library (nM)'],
        '% Deviation': non_standard_df['% Deviation'],
        'Concentration of undiluted library (ng/µL)': non_standard_df['Concentration of undiluted library (ng/µL)'],
        'Working concentration (pM)' : non_standard_df['Working concentration (pM)'],
        'Working concentration (nM)' : non_standard_df['Working concentration (nM)'],
        'Working concentration (ng/µL)' : non_standard_df['Working concentration (ng/µL)']
    })
    
    st.subheader("Calculate and review library concentrations:")
    st.dataframe(output_df)
    
    st.download_button(
        label="Download results as TSV",
        data=output_df.to_csv(sep='\t', index=False),
        file_name=f'{pcr_data_basename}_{dye_id}_KAPA_analysis_per_well_results.tsv',
        mime='text/tsv',
        key=f'{pcr_data_basename}_{dye_id}_KAPA_analysis_per_well'
        )

    # Create a summary DataFrame with one row per unique combination of 'SampleID' and 'dilutionfactor'
    summary_df = output_df.groupby(['SampleID', 'Dilution']).agg({
        'Working concentration (nM)': 'first',
        'Working concentration (pM)': 'first',
        'Working concentration (ng/µL)': 'first',
        'Concentration of undiluted library (pM)': 'first',
        'Concentration of undiluted library (nM)': 'first',
        'Concentration of undiluted library (ng/µL)': 'first',
        'Size-adjusted concentration (pM)': 'first',
        'Average fragment length (bp)': 'first',
        '% Deviation': 'first',
        'Average Cq': 'first',
        'Delta Cq': 'first',
        'log (concentration pM)': 'first',
        'Average concentration (pM)': 'first',
    }).reset_index()

    # Display the summary table
    st.subheader("KAPA Summary Table:")
    st.dataframe(summary_df)
    
    st.download_button(
        label="Download results as TSV",
        data=summary_df.to_csv(sep='\t', index=False),
        file_name=f'{pcr_data_basename}_{dye_id}_KAPA_analysis_per_sample_results.tsv',
        mime='text/tsv',
        key=f'{pcr_data_basename}_{dye_id}_KAPA_analysis_per_sample'
            )

def has_non_standard_wells(results_df):
    # Regular expression pattern for wells named "Standard_<n>"
    pattern = re.compile(r'^Standard_\d+$')  # ^ and $ ensure the whole string matches

    standard_rows = results_df['SampleID'].str.match(pattern)

    # Check if there are any rows that do NOT match the pattern
    non_standard_exists = not all(standard_rows)

    # Check if there are any wells left after filtering
    return non_standard_exists

def on_sample_change():
    st.rerun()

def handle_well_buttons(unique_identifiers, fixed_column_no, sorted_wells, identifier_type):
    # Generate the full set of identifiers
    if identifier_type == 'row':
        all_identifiers = [chr(i) for i in range(ord('A'), ord('P') + 1)]  # Generate list from 'A' to 'P'
    else:  # 'column'
        all_identifiers = [f"{i:02d}" for i in range(1, 25)]  # Generate zero-filled list from '01' to '24'

    button_cols = st.columns(fixed_column_no)
    
    # Find the maximum label length for consistent button sizes
    max_label_length = max(len(identifier) for identifier in all_identifiers)

    # Iterate over all possible identifiers
    for i, identifier in enumerate(all_identifiers):
        with button_cols[i % fixed_column_no]:
            key = f"{identifier_type}_{identifier}"  # Unique key for each button
            # For columns, check both zero-filled and non-zero-filled formats to ensure compatibility
            if identifier in unique_identifiers or (identifier_type == 'column' and identifier.lstrip('0') in unique_identifiers):
                # Create a functional button for identifiers in the list
                button_label = identifier  # Use the zero-filled identifier as the label
                if st.button(button_label, key=key, on_click=lambda: st.session_state.update({'plot_and_calculate': False})):
                    if identifier_type == 'row':
                        handle_row_button(identifier, sorted_wells)
                    else:
                        # For column handling, strip leading zeros
                        handle_column_button(identifier.lstrip('0'), sorted_wells)
            else:
                # Create a placeholder with non-breaking spaces to match the size
                placeholder_label = "\u00A0" * max_label_length
                st.button(placeholder_label, key=f"placeholder_{key}", disabled=True)

    # Additional empty space for alignment
    extra_space = fixed_column_no - len(all_identifiers) % fixed_column_no
    for _ in range(extra_space if extra_space != fixed_column_no else 0):
        with button_cols[_]:
            st.empty()  # Maintains alignment with an empty block
            
# Function to toggle a section and turn off others
def toggle_section(active_key):
    # Toggle the state of the active section
    current_state = not st.session_state.get(active_key, False)
    # Set all sections to False, then apply the current state to the active section
    for key in ['add_by_row', 'add_by_column', 'add_by_sample', 'add_by_well']:
        st.session_state[key] = False
    st.session_state[active_key] = current_state
    st.session_state['plot_and_calculate'] = False
    st.rerun()

# Function to read the CSV file line by line and return as a list of lines
def read_tsv_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line)
    return lines

# Function to download the list of lines as a text file
def download_tsv(lines):
    # Create a button to download the data as a plain text file
    txt_data = "".join(lines)
    st.sidebar.download_button(
        label="Download labelling template",
        data=txt_data,
        key="download-button",
        file_name="CyclusK_qPCR_labelling_template_384_well_including_KAPA_standards.tsv"
    )
    
section_keys = ['add_by_row', 'add_by_column', 'add_by_sample', 'add_by_well', 'plot_and_calculate', 'selected_wells', 'multiselect_interaction', 'color_by_samples']
for key in section_keys:
    if key not in st.session_state:
        st.session_state[key] = False
        
if 'previous_selected_wells' not in st.session_state:
    st.session_state['previous_selected_wells'] = []

def main():   
    st.set_page_config(page_title='Cyclus Κ', page_icon='CyclusK_logo.ico')

    col1, col2 = st.columns([1, 3.1])

    with col1:
        st.title('Cyclus Κ')
       
    with col2:
        st.image('CyclusK_logo.svg')
        
    st.subheader("qPCR and KAPA Library Quantification Analysis Tool")
    uploaded_PCR_file, uploaded_labelling_file, pcr_data_basename = upload_files()
      
    labelling_data = process_labelling_file(uploaded_labelling_file)
    
    raw_data, melt_data = process_pcr_data_file(uploaded_PCR_file, pcr_data_basename)
    raw_data, melt_data = merge_and_preprocess_data(raw_data, melt_data, labelling_data, pcr_data_basename)
    subtracted_data = pd.DataFrame()

    if st.session_state['selected_wells'] != st.session_state['previous_selected_wells']:
        st.session_state['multiselect_interaction'] = True
        st.session_state['previous_selected_wells'] = st.session_state['selected_wells']
    else:
        st.session_state['multiselect_interaction'] = False

    if uploaded_PCR_file:
        st.button('Plot and perform calculations', on_click=lambda: st.session_state.update({'plot_and_calculate': True}) and st.session_state.update({'multiselect_interaction': False}))
        if not st.session_state['plot_and_calculate']:
                st.markdown('**Note**: <span style="margin-left: 0.5em;">Performing plotting and calculations on a large number of wells can take some time.</span><br><span style="margin-left: 3em;">Adjust parameters on a small subset before running your final analysis.</span>', unsafe_allow_html=True)
    else:
        st.write("Upload your qPCR run file in the sidebar.")
        
    if raw_data is not None :
        if st.session_state['selected_wells']:
            unique_dyes = setup_analysis_parameters(raw_data)

            if st.session_state['plot_and_calculate'] and not st.session_state['multiselect_interaction']:
                subtracted_data, baseline_end_cycles = baseline_subtraction(raw_data)
            
            if not subtracted_data.empty:
                for dye in unique_dyes:
                    results_df, all_steepest_sections, cq_thresholds = analyze_data(subtracted_data, unique_dyes, labelling_data)
                    results_df['Cq'] = pd.to_numeric(results_df['Cq'], errors='coerce')
                                       
                plot_amplification_curves(subtracted_data, cq_thresholds, unique_dyes, all_steepest_sections, baseline_end_cycles)

                if not melt_data.empty:
                    melt_df = melt_data[melt_data['WellID'].isin(st.session_state['selected_wells'])].copy()
                    melt_temperatures_df = plot_melt_curves(melt_df, unique_dyes, prominence=0.01)
                    results_df = pd.merge(results_df, melt_temperatures_df, on=['DyeID', 'WellID'], how='left')
                    
                display_results_table(results_df, labelling_data, pcr_data_basename)

        current_data = raw_data if raw_data is not None else subtracted_data

        # Define sorted_wells
        available_wells = sorted(current_data['WellID'].unique())
        sorted_wells = sort_wells(available_wells)  # Assuming sort_wells is a function that sorts well identifiers
        
        if st.session_state['selected_wells']:
            st.session_state['selected_wells'] = [well for well in st.session_state['selected_wells'] if well in sorted_wells]
        else:
            st.session_state['selected_wells'] = []

        # Plot standard curves if standards data is available
        if not subtracted_data.empty and labelling_data is not None and st.session_state['selected_wells']:
            standards_data = results_df[results_df['sampletype'] == 'standard'].copy()
            standards_data['Cq'] = pd.to_numeric(standards_data['Cq'], errors='coerce')
            standards_data['std_conc_pm'] = pd.to_numeric(standards_data['std_conc_pm'], errors='coerce')
            standards_data.sort_values(by='std_conc_pm', inplace=True)

            # Filter standards_data for only selected wells
            standards_data = standards_data[standards_data['WellID'].isin(st.session_state['selected_wells'])]

            # Group by 'std_conc_pm' and 'DyeID' and calculate the average Cq for each group
            avg_cq_data = standards_data.groupby(['std_conc_pm', 'DyeID'])['Cq'].mean().reset_index()

            for dye_id in unique_dyes:         
                fig, ax = plt.subplots()
                dye_standards_data = standards_data[standards_data['DyeID'] == dye_id]
                dye_avg_cq_data = avg_cq_data[avg_cq_data['DyeID'] == dye_id]
                if not dye_avg_cq_data.empty and has_non_standard_wells(results_df):
                    # Perform linear regression and get regression line
                    linreg, r_squared, log_std_conc_pm = perform_linear_regression(dye_avg_cq_data)
                    x_values = np.linspace(log_std_conc_pm.min(), log_std_conc_pm.max(), 100)
                    y_values = linreg.predict(x_values.reshape(-1, 1))
                    slope, intercept = plot_standard_curve(ax, dye_standards_data, dye_id, (x_values, y_values, linreg))

                    # Show the plot
                    st.pyplot(fig)
                    
                    if has_non_standard_wells(results_df):
                        generate_standard_curve_summary(dye_standards_data, pcr_data_basename, dye_id)
                    
                        if st.session_state['kapa_analysis']:
                            calculate_qpcr_results(results_df, slope, intercept, pcr_data_basename, dye_id)

        st.subheader("Select Wells to Include:")

        if st.button("Select All Wells", on_click=lambda: st.session_state.update({'plot_and_calculate': False})):
        
        
            if not all(well in st.session_state['selected_wells'] for well in sorted_wells):
                st.session_state['selected_wells'] = sorted_wells
            else:
                st.session_state['selected_wells'] = []
            st.rerun()

        # Button and section for "Add by Well"
        if st.button('Add or Remove by Individual Wells:', key='toggle_wells', on_click=lambda: st.session_state.update({'plot_and_calculate': False})):
            toggle_section('add_by_well')
        if st.session_state.get('add_by_well'):
            # Assuming 'sorted_wells' is defined
            selected_wells = st.multiselect(
                "Select Wells",
                options=sorted_wells,
                key='selected_wells'
            )

        sample_ids = current_data['SampleID'].unique()
        sample_id_to_color = generate_sample_id_to_color(sample_ids, None)

        # Assuming 'current_data' has columns 'WellID' and 'SampleID'
        well_id_to_sample_id = dict(zip(current_data['WellID'], current_data['SampleID']))
        if 'sampletype' in current_data:
            well_id_to_sampletype = dict(zip(current_data['WellID'], current_data['sampletype']))
        else:
            well_id_to_sampletype = {}

        # Now, call generate_well_table with the required arguments
        generate_well_table(sorted_wells, sample_id_to_color, well_id_to_sample_id, well_id_to_sampletype)

        # Extract unique rows and columns from the dataset
        unique_rows = sorted(set(re.match(r"([A-Za-z]+)", well).group(1) for well in available_wells))
        unique_columns = sorted(set(re.match(r"[A-Za-z]+([0-9]+)", well).group(1).zfill(2) for well in available_wells))

        # Button and section for "Add by Row"
        if st.button('Add or Remove Wells by Rows:', key='toggle_rows', on_click=lambda: st.session_state.update({'plot_and_calculate': False})):
            toggle_section('add_by_row')
        if st.session_state.get('add_by_row'):
            # Call the function for rows
            handle_well_buttons(unique_rows, 16, sorted_wells, 'row')
        # Button and section for "Add by Column"
        if st.button('Add or Remove Wells by Columns:', key='toggle_columns', on_click=lambda: st.session_state.update({'plot_and_calculate': False})):
            toggle_section('add_by_column')
        if st.session_state.get('add_by_column'):
            # Call the function for columns
            handle_well_buttons(unique_columns, 12, sorted_wells, 'column')
        # Button and section for "Add by Sample"
        if st.button('Add or Remove Wells by Samples:', key='toggle_samples', on_click=lambda: st.session_state.update({'plot_and_calculate': False})):
            toggle_section('add_by_sample')
        if st.session_state.get('add_by_sample'):
            # Assuming 'current_data' and 'available_wells' are defined
            sample_ids = current_data['SampleID'].unique()
            display_sample_buttons(sample_ids, current_data, available_wells)
                                            
if __name__ == "__main__":
    main()
