Getting Started
-----------------

Below we present a simplified training routine where we focus on the lines that create stages and inject fixtures:

.. code-block::
   :linenos:

   class TrainManager(Extensible):

     def train(...):

       with self.staged(
           "train", {"epoch_num": 0}
       ):
         # Train
         for _ in range(self.fixtures["epoch_num"], self.epochs):

           with self.staged("train_step_epoch"):

             for batch in ... :
               with self.staged(
                   "train_step_batch",
                   {
                       "batch": batch,
                       "true_batch_size": self.get_true_batch_size(batch),
                   },
               ):
                 ...

                 prediction = self.model.forward(batch)
                 self.fixtures["prediction"] = prediction

                 loss = self.loss(batch, prediction)
                 self.fixtures["loss"] = loss

                 ...

             self.fixtures.modify("epoch_num", self.fixtures["epoch_num"] + 1)


* **train**: *epoch_num*

  * **train_step_epoch**

    * **train_step_batch**: *batch*, *true_batch_size*, *prediction*, *loss*
