#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Task 1 - 1
# TIES 4911
# Toni Pikkarainen
# 14.1.2020
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[2]:


# Constants
# Lecture02, slides 11-13
a = tf.constant(5)
b = tf.constant(2)
add_op = tf.add(a, b)
mul_op = tf.multiply(b,add_op)
pow_op = tf.pow(mul_op,b)







# In[3]:


# Variables 
# Lecture02, slides 15-17
var1 = tf.Variable(2, name="scalar1")
var2 = tf.Variable(3, name="scalar2")
assign_op = var2.assign(10)


# In[4]:


# Placeholders
# Lecture02, slides 19-20

a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([1, 2, 3], tf.float32)
d = tf.constant([2, 3, 4], tf.float32)
c = tf.add(a, tf.add(d,b))



# In[5]:


list_of_values = [1,2,3]


# In[6]:


# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(var1.initializer)
    
    # Question:
    # If I add this next line, everytime I run the code tensorboard will
    # print new assign_op node for var1 - how can I prevent that from happening??
    # SOLVED: restart kernel!
    
    # Why these are not producing op_nodes to tensorboard graph?
    sess.run(var1.assign_add(var1.eval()))
    sess.run(var1.assign_sub(3))
    print(var1.eval())
    
    print(sess.run(pow_op))
    sess.run(assign_op)
    print(var2.eval())
    print (sess.run(c, {a: [1, 1, 1]}))
    writer.close() 


# In[ ]:




