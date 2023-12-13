
Train Manager: Extensible ML methods
---------------------------------------------

Train Manager addresses a common organization issue in machine learning routines where the algorithm, visualization and book keeping code are intertwined in the same code block, making it difficult to add new components. The main reason for this is the need to address variables that occur at specific places in the routine (e.g., the batch, prediction or loss).

Train Manager addresses this by means of **extensible methods** that are structured into nested stages. Extensions can be added to the callable in question without modifying its code by declaring classes with hook methods that have special names specifying whether they will execute before or after a given stage.

In order to provide an intuitive access to the variables inside the different stages of an extensible methos, Train Manager employs a system of **fixtures** inspired on `pytest <https://docs.pytest.org/>`_ that enables programers to access internal variables of the extensible method designated as fixtures simply by using their name in the hook's signature. The approach also manages the lifetime of these fixtures and prevents them from being accidentlay redefined based on the lifetime of the stage where the fixture was first defined, thus avoiding silent bugs due to fixture leakages.
