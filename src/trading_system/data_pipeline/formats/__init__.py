"""Vendor file formats: one reader per layout, none of them interpreting units.

A reader's job ends at "these are the rows the vendor wrote". Deciding what the
volume column means, what timezone the stamps are in, and what to do about
duplicates belongs to normalisation, which sees every format at once and can
therefore be consistent about it.

Nothing is re-exported here. There is one reader so far and its surface is four
names; importing them from the module that defines them keeps the format a
caller is reading visible at the import site.
"""
