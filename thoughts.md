# Independent Groups Search:

The suggested configuration should be:
1. untested, and not currently in testing, within its group
2. a _possible_ best of the group
    - descendant of the best config in the preceding group (prioritized)
    - descendant or an untested config
3. have the minimum group number out of all configuration satisfying the above 

A sub-searcher should be discarded if it is _guaranteed_ to be invalid, i.e. a
configuration in the prior group was found to be better than the parent configuration.

Parent configurations should _only_ be leaf nodes.