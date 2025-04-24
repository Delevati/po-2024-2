# Beamer Presentation

This project is a LaTeX Beamer presentation designed to showcase research findings and related content. Below are the details of the project structure and instructions for compiling the presentation.

## Project Structure

- **images/**: Contains images used in the presentation.
  - `areas.png`: An image that can be included in the slides.

- **bibliography/**: Contains bibliography entries.
  - `references.bib`: A BibTeX file with references that can be cited in the presentation.

- **slides.tex**: The main LaTeX source file for the Beamer presentation. This file defines the structure of the slides, including sections, frames, and content.

- **beamerthemeCustom.sty**: A custom Beamer theme file that allows for personalized styling and layout of the presentation.

- **README.md**: This documentation file, which provides an overview of the project and instructions for use.

## Compiling the Presentation

To compile the presentation, follow these steps:

1. Ensure you have a LaTeX distribution installed (e.g., TeX Live, MiKTeX).
2. Open a terminal and navigate to the project directory.
3. Run the following command to compile the `slides.tex` file:

   ```
   pdflatex slides.tex
   ```

4. If you have citations in your presentation, run BibTeX:

   ```
   bibtex slides
   ```

5. Compile the `slides.tex` file again twice to ensure all references are updated:

   ```
   pdflatex slides.tex
   pdflatex slides.tex
   ```

## Dependencies

Make sure you have the following packages installed in your LaTeX distribution:

- `beamer`
- `graphicx`
- `biblatex` (if using BibTeX for references)

## Customization

You can customize the appearance of your presentation by modifying the `beamerthemeCustom.sty` file. This file allows you to change colors, fonts, and layout settings to fit your preferences.

## License

This project is licensed under the MIT License. Feel free to modify and use it as needed.