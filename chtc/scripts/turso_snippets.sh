# Install
curl -sSfL https://get.tur.so/install.sh | bash

# Login
turso auth login --headless

# Dump from Turso's database
turso db shell usgs-mineral .dump > ./tmp/dump_mineral_240614.sql

# Restore a database from a dump
sqlite3 ./tmp/mineral_data_240614.sqlite3 < ./tmp/dump_mineral_240614.sql
