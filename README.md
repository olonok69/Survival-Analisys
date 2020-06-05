# SurVival regression  and AFI HTTP restful Interface

Asset failure Interface Deep Survival regression model

Survival regression / Cox's proportional Hazard model (lifetimes)
https://readthedocs.org/projects/lifetimes/downloads/pdf/latest/

Model selection based on predictive power
If censoring is present, it’s not appropriate to use a loss function like mean-squared-error or mean-absolute-loss. Instead, one measure is the concordance-index, also known as the c-index. This measure evaluates the accuracy of the ranking of predicted time. It is in fact a generalization of AUC, another common loss function, and is interpreted similarly:
0.5 is the expected result from random predictions,
1.0 is perfect concordance and,
0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)


# Deep Survival Neural Network (DSNN)

Loss Function: negative log of Breslow Approximation partial likelihood function
https://en.wikipedia.org/wiki/Proportional_hazards_model


See docum 
def _create_loss(self):
"""
      Define the loss function.
      Notes
      -----
      The negative log of Breslow Approximation partial
      likelihood function. See more in "Breslow N.. See more in
      "Breslow N., 'Covariance analysis of censored
      survival data, ' Biometrics 30.1(1974):89-99.".
      """
      with tf.name_scope("loss"):
      # Obtain T and E from self.Y
      # NOTE: negtive value means E = 0
      Y_c = tf.squeeze(self.Y)
      Y_hat_c = tf.squeeze(self.Y_hat)
      Y_label_T = tf.abs(Y_c)
      Y_label_E = tf.cast(tf.greater(Y_c, 0), dtype=tf.float32)
      Obs = tf.reduce_sum(Y_label_E)
      Y_hat_hr = tf.exp(Y_hat_c)
      Y_hat_cumsum = tf.log(tf.cumsum(Y_hat_hr))
      # Start Computation of Loss function
      # Get Segment from T
      unique_values, segment_ids = tf.unique(Y_label_T)
      # Get Segment_max
      loss_s2_v = tf.segment_max(Y_hat_cumsum, segment_ids)
      # Get Segment_count
      loss_s2_count = tf.segment_sum(Y_label_E, segment_ids)
      # Compute S2
      loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v,
      loss_s2_count))
      # Compute S1
      loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
      # Compute Breslow Loss
      loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)
      # Compute Regularization Term Loss
      reg_item =
      tf.contrib.layers.l1_l2_regularizer(self.config["L1_reg"],
      self.config["L2_reg"])
      loss_reg = tf.contrib.layers.apply_regularization(reg_item,
      tf.get_collection("var_weight"))
      # Loss function = Breslow Function + Regularization Term
      self.loss = tf.add(loss_breslow, loss_reg)
