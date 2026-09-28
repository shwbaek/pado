Frequency-domain Light
======================

.. py:module:: pado.kspace_light

.. autoclass:: KSpaceLight
   :members:
   :exclude-members: get_intensity, __weakref__

.. py:method:: KSpaceLight.get_intensity(c=None)

   Return ``field.abs().square()`` for all channels, or for channel ``c``.

   :param c: Optional channel index.
   :rtype: torch.Tensor
