#!/usr/bin/env bash


# Variables
blue_color='\033[0;34m'  # ANSI blue color.
reset_color='\e[0m'      # ANSI default color.
venv_path='./.venv'
venv_subdir=$([[ -d /c/Windows ]] && echo 'Scripts' || echo 'bin')
venv_activator_path="${venv_path}/${venv_subdir}/activate"
migrations_path="$(pwd)/database/migrations"


#########
# Adds or removes a database migration.
#########
# Usage:
## migrate add <migration_name>
## migrate apply
## migrate remove
#########
## migration_name (string): The name of the migration.
#########
migrate()
{
  # Require at least one argument.
  if [ $# -lt 1 ]; then
    echo 'Usage: migrate {add|apply|rm|remove} [args]' >&2
    return 1
  fi

  if [[ ! -d $migrations_path ]]; then
    if aerich init-db; then
      echo 'Initial migration successfully created. Exiting...'
      return 0
    else
      echo 'Failed to initialize the database. Please, delete the migrations folder and all database tables then try again.' >&2
      return 1
    fi
  fi

  case "$1" in
    add)
      # For the "add" command, the second argument (migration name) is mandatory.
      if [[ -z $2 ]]; then
        echo "Error: 'add' requires a migration name as second argument." >&2
        return 1
      fi

      local migration_file
      migration_file=$(aerich migrate --no-input --name "$2" | grep -oP '\d+_\d+_\S+\.py')

      # "migration_file" will be empty/unset if grep fails to match the regex.
      if [[ -z $migration_file ]]; then
        echo 'Project models are already up-to-date, no migration is needed.' >&2
        return 1
      fi

      echo -e "Migration ${blue_color}${migration_file}${reset_color} created successfully."
      echo -e "To apply the changes to the database, execute ${blue_color}migrate apply${reset_color}"
      ;;

    apply)
      if aerich upgrade --in-transaction true
      then
          echo -e "To revert the changes, execute ${blue_color}migrate remove${reset_color}"
      else
          echo 'An error has occurred, please review or revert the problematic migration.' >&2
      fi
      ;;

    rm|remove)
      local migration_file
      migration_file=$(aerich downgrade -d --yes >&1 | grep -oP '\d+_\d+_\S+\.py')

      # "migration_file" will be empty/unset if grep fails to match the regex.
      if [[ -z $migration_file ]]; then
        echo "There are no migrations to be removed." >&2
        return 1
      else
        echo -e "Removed migration ${blue_color}${migration_file}${reset_color}."
      fi
      ;;

    *)
      echo "Error: First argument must be 'add', 'apply', 'rm', or 'remove'." >&2
      return 1
      ;;
  esac
}

#########
# Shorthand for deactivating the virtual environment
#########
# Usage: quit
#########
quit()
{
  deactivate
  unset migrate
  unset quit
}


# Main Entry Point
if [[ ${BASH_SOURCE-} == "$0" ]]; then
  echo -e "You must source this script: ${blue_color}source $0${reset_color}" >&2
  exit 1
fi

# Create the virtual environment if it doesn't exist
if [[ ! -f $venv_activator_path ]]; then
  python -m venv $venv_path
fi

# Activate the virtual environment
source "$venv_activator_path"
pip install -e .[dev]
rm -rf ./loom_backend.egg-info

echo -e '\nVirtual environment activated.'
echo -e "To execute the application in development mode, run ${blue_color}fastapi dev${reset_color}"
echo -e "To execute the application in production mode, run ${blue_color}fastapi run${reset_color}"
echo -e "To exit the virtual environment, run ${blue_color}quit${reset_color}"
