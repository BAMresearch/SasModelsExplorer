# SasModels Explorer App

A PyQt-based interactive application to explore scattering models using the sasmodels library. 

<img width="1010" alt="image" src="https://github.com/user-attachments/assets/cf9931d7-570b-4e46-9521-9c13bd190493">

This app enables visualization and fine-tuning of model parameters via a user-friendly GUI, featuring both sliders and dropdown menus for parameter adjustments, and a real-time plot of scattering intensity.

## Features

- **Interactive Parameter Controls**: Adjust model parameters with sliders, text boxes, and dropdowns.
- **Real-Time Plotting**: View scattering intensity plots that update in real time as parameters are adjusted.
- **Polydispersity Support**: Automatically includes polydispersity controls for relevant parameters.
- **Custom Model Loading**: Load any supported `sasmodels` model by typing its name and pressing Enter.
- **Command-Line Interface (CLI)**: Run the app or interact with models directly via the command line.

Here's an updated `README.md` with a streamlined installation section referencing `requirements.txt`, and details on the command-line interface (CLI) functionality based on the presence of `__main__.py`.

---

## SasModels Explorer App

A PyQt-based interactive application for exploring scattering models using the `sasmodels` library. This app offers a GUI with sliders, text boxes, and dropdowns for fine-tuning parameters, displaying real-time scattering intensity plots as parameters are adjusted.

### Features
- **Interactive Parameter Controls**: Adjust model parameters with sliders, text boxes, and dropdowns.
- **Real-Time Plotting**: View scattering intensity plots that update in real time as parameters are adjusted.
- **Polydispersity Support**: Automatically includes polydispersity controls for relevant parameters.
- **Custom Model Loading**: Load any supported `sasmodels` model by typing its name and pressing Enter in the text field or when launching from the command line.
- **Command-Line Interface (CLI)**: Run the app or interact with models directly via the command line.

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Requirements
Dependencies are managed in `requirements.txt`, and include:
   - `logging`
   - `matplotlib`
   - `numpy`
   - `PyQt5`
   - `sasmodels`
   - `pint` for future unit conversions

### Usage

#### Running the GUI
To launch the GUI, execute:
```bash
python /path/to/SasModelsExplorer/ModelExplorer -v sphere@hardsphere
```

### Using the GUI

1. **Loading a Model**: 
   - Type the model name (e.g., `"sphere"`, `"cylinder"`, or sasmodels-syntax combinations such as `"sphere@hardsphere+porod"`) in the `Model` field and press **Enter** to load it.
   - The app will display parameter controls for the model.

2. **Adjusting Parameters**:
   - **Sliders and Text Boxes**: Each parameter has an adjustable slider and a text box for exact values. Logarithmic sliders enable wide-range adjustments.
   - **Dropdown Menus**: Parameters with multiple text options (e.g., `_pd_type` for polydispersity) appear as dropdowns.

3. **Polydispersity Controls**:
   - For parameters supporting polydispersity, additional controls for `_pd` (polydispersity level) and `_pd_type` (distribution type) are added automatically.

4. **q-Range Adjustment**:
   - Adjust the `Q Min` and `Q Max` fields below the plot to control the q-range of the scattering plot.

### Code Overview

#### Key Classes and Methods
- **`SasModelApp`**: Main application class.
  - **`load_model_parameters`**: Loads parameters for the selected model, dynamically generating controls based on the model's parameters.
  - **`create_parameter_input_element`**: Generates an input element (slider, text box, or dropdown) for each parameter.
  - **`update_plot`**: Computes the scattering intensity based on current parameters and updates the plot.
- **Polydispersity Parameters**: Automatically adds `_pd` and `_pd_type` controls for parameters that support polydispersity.

### Example Models
Try these models to get started:
- `"sphere"`: Spherical scatterer model.
- `"cylinder"`: Cylindrical scatterer model.
- `"sphere@hardsphere"`: sphere model with a hard-shell structure factor.
- `"sphere@hardsphere+porod"`: sphere model with a hard-shell structure factor and additional Porod slope.

The entire sasmodels library is available, which you can combine, multiply, subtract with the syntax alluded to above... A help text with the available models is displayed when a nonexistent model is entered.

### Logging and Debugging
The application uses `logging` for tracking parameter changes:
- Logs are printed in the console to show current parameter values.
- To enable detailed logs, set the logging level to `DEBUG` in the code or use the -vv option at the CLI. 

### Current issues
Also check the "issues" in the Github repository.
- The app currently does not display 2D scattering models. They will probably not be fast enough to be fun anyway
- There is an issue clearing the parameters when entering a new model, causing the new parameters to be shifted down in the UI. 
- There should be a timeout for model calculations that take long.
- The font in the help dialog is not monospaced, which makes it hard to read.

### Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

### License
This project is licensed under the MIT License.

---
