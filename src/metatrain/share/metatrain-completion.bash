_metatrain()
{
  local cur_word="${COMP_WORDS[$COMP_CWORD]}"
  local prev_word="${COMP_WORDS[$COMP_CWORD-1]}"
  local module="${COMP_WORDS[1]}"

  # Define supported file endings.
  local yaml='!*@(.yml|.yaml)'
  local ckpt='!*.ckpt'
  local pt='!*.pt'
  local architecture_names=$(python -c "
from metatrain.utils.architectures import find_all_architectures
print(' '.join(find_all_architectures()))
  ")

  # Complete the arguments to the module commands.
  case "$module" in
    train)
      case "${prev_word}" in
        -h|--help|-o|--output|-r|--override)
          COMPREPLY=( )
          return 0
          ;;
        -c|--continue)
          COMPREPLY=( $( compgen -f -X "$ckpt" -- "${cur_word}") )
          return 0
          ;;
        *)
          if [[ $COMP_CWORD -eq 2 ]]; then
            COMPREPLY=( $(compgen -f -X "$yaml" -- "${cur_word}") )
            return 0
          fi
          ;;
      esac
      local opts="-h --help -o --output -c --continue -r --override"
      COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
      return 0
      ;;
    export)
      case "${prev_word}" in
        -h|--help|-o|--output)
          COMPREPLY=( )
          return 0
          ;;
        *)
          if [[ $COMP_CWORD -eq 2 ]]; then
            # We don't have a generated list of known the architecture names
            COMPREPLY=( $(compgen -W "$architecture_names" -- "${cur_word}") )
            return 0
          elif [[ $COMP_CWORD -eq 3 ]]; then
            COMPREPLY=( $(compgen -f -X "$ckpt" -- "${cur_word}") )
            return 0
          fi
          ;;
      esac
      local opts="-h --help -o --output -m --metadata --token"
      COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
      return 0
      ;;
    eval)
      case "${prev_word}" in
        -h|--help|-o|--output|-b|--batch-size|--check-consistency)
          COMPREPLY=( )
          return 0
          ;;
        -e|--extensions-dir)
          # Only complete directories
          COMPREPLY=( $(compgen -d -- "${cur_word}") )
          return 0
          ;;
        *)
          if [[ $COMP_CWORD -eq 2 ]]; then
            COMPREPLY=( $(compgen -f -X "$pt" -- "${cur_word}") )
            return 0
          elif [[ $COMP_CWORD -eq 3 ]]; then
            COMPREPLY=( $(compgen -f -X "$yaml" -- "${cur_word}") )
            return 0
          fi
          ;;
      esac
      local opts="-h --help -o --output -b --batch-size -e --extensions-dir --check-consistency"
      COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
      return 0
      ;;
  esac

  # Complete the basic metatrain commands.
  local opts="eval export train -h --help --debug --version"
  COMPREPLY=( $(compgen -W "${opts}" -- "${cur_word}") )
  return 0
}

if test -n "$ZSH_VERSION"; then
  autoload -U +X compinit && compinit
  autoload -U +X bashcompinit && bashcompinit
fi

complete -o bashdefault -F _metatrain mtt
