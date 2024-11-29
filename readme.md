# SasModels Explorer App

A PyQt-based interactive application to explore scattering models using the `sasmodels` library.

![SME](https://github.com/user-attachments/assets/9495bd08-09a6-43a2-92a1-67d7cb48e1a8)

This app enables visualization and fine-tuning of model parameters via a user-friendly GUI, featuring both sliders and dropdown menus for parameter adjustments, and a real-time plot of scattering intensity.

## Features

- **Interactive Parameter Controls**: Adjust model parameters with sliders, text boxes, and dropdowns.
- **Real-Time Plotting**: View scattering intensity plots that update in real time as parameters are adjusted.
- **Polydispersity Support**: Automatically includes polydispersity controls for relevant parameters.
- **Custom Model Loading**: Load any supported `sasmodels` model by typing its name and pressing Enter.
- **Command-Line Interface (CLI)**: Run the app or interact with models directly via the command line.

## Installation and Usage

1. clone the repository: 
```bash
git clone https://github.com/SasView/explorer`
cd <repository folder>
```
2. (optional). setup a virtual enviroment for this explorer: `python3.12 -m venv .venv`
3. (optional). activate the virtual environment: `source .venv/bin/activate`
4. install the dependencies: `pip install -r requirements.txt`
5. run the app: `python -m ModelExplorer -v sphere@hardsphere`
6. (optional) deactivate the virtual environment: `deactivate`

## Using the GUI
### Loading a Model:

Type the model name (e.g., "sphere", "cylinder", or sasmodels-syntax combinations such as 
"sphere@hardsphere+cylinder") in the model field and press Enter to load it.
The app will display parameter controls for the model.

### Adjusting Parameters:

Sliders and Text Boxes: Each parameter has an adjustable slider and a text box for exact values. Logarithmic sliders enable wide-range adjustments.
Dropdown Menus: Parameters with multiple text options (e.g., _pd_type for polydispersity) appear as dropdowns.
Polydispersity Controls: Automatically includes controls for relevant parameters: polydispersity and (default number-weighted) distribution shape.
Axis controls: Under the graph, the q limits and q units can be adjusted. 

### Example Models
Try these models to get started:
- `"sphere"`: Spherical scatterer model.
- `"cylinder"`: Cylindrical scatterer model.
- `"sphere@hardsphere"`: sphere model with a hard-shell structure factor.
- `"sphere@hardsphere+porod"`: sphere model with a hard-shell structure factor and additional Porod slope.

The entire sasmodels library is available, which you can combine, multiply, subtract with the syntax alluded to above... A help text with the available models is displayed when a nonexistent model is entered.

## Under the hood
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
