# Interpret3C: Interpretable Student Clustering Through Individualized Feature Selection

This repository is the official implementation of the AIED 2024 LBR Paper entitled ["Interpret3C: Interpretable Student Clustering Through Individualized Feature Selection"]() written by [Isadora Salles](https://github.com/isadorasalles), [Paola Mejia-Domenzain](https://github.com/paola-md)*, [Vinitra Swamy*](http://github.com/vinitra), [Julian Blackwell](https://ch.linkedin.com/in/julian-blackwell-93407a13b), and [Tanja Käser](https://people.epfl.ch/tanja.kaeser/?lang=en).

## Overview

Clustering in education, particularly in large-scale online environments like MOOCs, is essential for understanding and adapting to diverse student needs. However, the effectiveness of clustering depends on its interpretability, which becomes challenging with high-dimensional data. Existing clustering approaches often neglect individual differences in feature importance and rely on a homogenized feature set. Addressing this gap, we introduce Interpret3C (Interpretable Conditional Computation Clustering), a novel clustering pipeline that incorporates interpretable neural networks (NNs) in an unsupervised learning context. This method leverages adaptive gating in NNs to select features for each student. Then, clustering is performed using the most relevant features per student, enhancing clusters' relevance and interpretability. We use Interpret3C to analyze the behavioral clusters considering individual feature importances in a MOOC with over 5,000 students. This research contributes to the field by offering a scalable, robust clustering methodology and educational case study that respects individual student differences and improves interpretability when using high-dimensional data.

## Usage guide

0. Install relevant dependencies with `pip install -r requirements.txt`.

1. Extract relevant features sets from MOOC courses (`BouroujeniEtAl`, `MarrasEtAl`, `LalleConati`, and `ChenCui`) through the ML4ED lab's EDM 2021 contribution on [benchmarks for feature predictive power](https://github.com/epfl-ml4ed/flipped-classroom). Place the results of these feature extraction scripts in `data/`.
  
2. Run your desired experiment from `notebooks/` by executing the notebook with Python 3.7 or higher.

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know. Feel free to file issues and pull requests on the repo and we will address them as we can.

## Citations
If you find this code useful in your work, please cite our paper:

```
Salles, I., Mejia-Domenzain, P., Swamy, V., Blackwell, J., Käser, T. (2024). 
Interpret3C: Interpretable Student Clustering Through Individualized Feature Selection. 
In: Proceedings of the 25th International Conference on Artificial Intelligence in Education. 
```

## License
This code is free software: you can redistribute it and/or modify it under the terms of the [MIT License](LICENSE).

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the [MIT License](LICENSE) for details.


