# Dump from Turso's database
turso db shell usgs .dump > ./tmp/dump_240508.sql

# Restore a database from a dump
sqlite3 ./tmp/data_240508.sqlite3 < ./tmp/dump_240508.sql
