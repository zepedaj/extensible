
Train Manager: Extensible ML methods
---------------------------------------------

Train Manager enables better *code organization* of *machine learning training experiments* by means of **extensible methods** and **fixtures**, making it possible to create reusable modules with common functionality like the built-in training, visualization, model graph debugging, and reproducibility tracking modules.

Extensible methods
    *Extensible methods* are Python methods that can be extended by injecting code (i.e., calling *extensions methods*) before and after pre-defined stages

Fixtures
    Fixtures are scoped variables that exist during a given stage and are available to extension methods called during that stage

.. code-block::

    class MyExtension(Extension):

        def pre_stage(self, var_a):
            print('Before the stage')

        def post_stage(self, var_a, var_b):
            print('After the stage: {var_a}, {var_b}')


    class MyExtensibleClass(Extensible):

        extensions = [MyExtension()]

        def my_extensible_method(self):
            with staged('my_stage', {'var_a':1}):
                self.fixtures['var_b'] = 2

In the example above, calling ``MyExtendedClass.my_extensible_method`` will call ``MyExtension.pre_stage`` and ``MyExtension.post_stage`` before and after the staged context code ``'my_stage'``, automatically passing in the variable values defined within stage `my_stage'



A common problem in training code is that the algorithm, visualization and book keeping code are intertwined in the same code block, making it difficult to add new components. The main reason for this is the need to address variables that occur at specific places in the routine (e.g., the batch, prediction or loss).

Train Manager addresses this by means of **extensible methods** that are structured into nested stages. Extensions can be added to the callable in question without modifying its code by declaring classes with hook methods that have special names specifying whether they will execute before or after a given stage.

In order to provide an intuitive access to the variables inside the different stages of an extensible methos, Train Manager employs a system of **fixtures** inspired on `pytest <https://docs.pytest.org/>`_ that enables programers to access internal variables of the extensible method designated as fixtures simply by using their name in the hook's signature. The approach also manages the lifetime of these fixtures and prevents them from being accidentlay redefined based on the lifetime of the stage where the fixture was first defined, thus avoiding silent bugs due to fixture leakages.


Organizing ML methods in this way makes it possible to more easily share common tasks across different project.

* Visualization tasks including

 * plotting the loss across epochs, batches or samples

 * visualizing the gradients of the various layers

* common debugging tasks such as

  * verifying that there is no leakage across examples in a batch in the forward/backawards passes due to errors in data manipulation (e.g., in using the correct parameters when calling torch.view, torch.reshape, torch.concatenate, torch.stack, etc)

* and reproducibility tasks like

  * keeping track of the exact commit and relative code differences of the current experiment

  * keeping track of the parameters used to instantiate the model (and other stateful objects) before loading a checkpoint -- and potentially modifying some of these like a threshold value for evaluation purposes.

All these tasks are the same across projects, and Train Manager includes out-of-the-box extensions to do all of these by simply adding each extension to a :class:`TrainManager` object.

The training routine and evaluation routines are further very repetitive across various projects, so being able to reuse the same routines easily while is another important benefit that allows researchers to focus on the model configurations and visualizations as opposed to rewriting a monotonous train routine, running the risk of introducing small bugs in the process.
