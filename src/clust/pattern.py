import json
import argparse
import os
import inflect

inflect = inflect.engine()

n2v = {
    'Ecology': 'described',
    'Reproduction': 'reproduce',
}
what_nouns = ['Species', 'Subspecies', 'Feeding']
where_nouns = ['Habitat', 'Distribution', 'Biology']
when_nouns = ['Life cycle', 'Description']
who_nouns = ['Ecology']
how_nouns = ['Reproduction']


def get_predicates(title):
    if inflect.singular_noun(title) is False:
        predicates = {'be': 'is', 'do': 'does', 'was': 'was'}
    else:
        predicates = {'be': 'are', 'do': 'do', 'was': 'were'}

    return predicates


def fill_pattern(lead_section, title):
    pattern = f"{lead_section} of {title}:"
    if lead_section in what_nouns:
        # What are the Species of Arctic fox?
        if lead_section == 'Feeding':
            pattern = f"What {get_predicates(title)['do']} the {title} feed on?"
        else:
            pattern = f"What {get_predicates(lead_section)['be']} the {lead_section} of {title}?"
    elif lead_section in where_nouns:
        if lead_section == 'Biology':
            # Where was Arctic fox recorded?
            pattern = f"Where {get_predicates(title)['was']} {title} recorded?"
        else:
            # Where does Arctic fox live?
            pattern = f"Where {get_predicates(title)['do']} {title} live?"
    elif lead_section in when_nouns:
        # When was Arctic fox described?
        pattern = f"When {get_predicates(title)['was']} {title} described?"
    elif lead_section in who_nouns:
        # Who described Arctic fox?
        pattern = f"Who {n2v[lead_section]} {title}?"
    elif lead_section in how_nouns:
        # How does Arctic fox reproduce?
        pattern = f"How {get_predicates(title)['do']} {title} {n2v[lead_section]}?"
    else:
        pattern = f"This document is about {title}:"

    return pattern


def select_clusts(args):
    if 'last_noisy' in args:
        last_noisy = args.last_noisy
    else:
        last_noisy = False

    if args.category == 'animal':
        if args.lead_section_num == 20:
            if args.topic_num == 4:
                # this is tailored for POWER
                animal_clusts = [
                    ['Taxonomy', 'Species', 'Subspecies', 'Classification'],
                    ['Distribution', 'Habitat'],
                    ['Description', 'Ecology', 'Behaviour', 'Biology', 'Diet', 'Feeding', 'Breeding',
                     'Reproduction'],
                    ['Status', 'Conservation'],
                ]
            elif args.topic_num == 2:
                animal_clusts = [
                    ['Distribution', 'Taxonomy', 'Species', 'Subspecies', 'Classification',
                     'Life cycle', 'Status', 'Conservation', 'Conservation status'],
                    ['Description', 'Habitat', 'Ecology', 'Behaviour', 'Biology', 'Diet', 'Feeding', 'Breeding',
                     'Reproduction'],
                ]
            else:
                animal_clusts = [
                    ['Distribution', 'Taxonomy', 'Species', 'Subspecies', 'Classification',
                     'Life cycle', 'Status', 'Conservation', 'Conservation status', 'Description', 'Habitat', 'Ecology',
                     'Behaviour', 'Biology', 'Diet', 'Feeding', 'Breeding',
                     'Reproduction']
                ]
        else:
            # 10 lead_section_num
            if args.topic_num == 4:
                if last_noisy:
                    animal_clusts = [
                        ['Distribution', 'Ecology'],
                        ['Description', 'Behaviour'],
                        ['Conservation', 'Habitat'],
                        ['Species', 'Subspecies'],
                        ['Status']
                    ]
                else:
                    animal_clusts = [
                        ['Distribution'],
                        ['Taxonomy', 'Species', 'Subspecies'],
                        ['Description', 'Habitat', 'Diet', 'Behaviour', 'Breeding'],
                        ['Conservation status'],
                    ]
            elif args.topic_num == 2:
                animal_clusts = [
                    ['Distribution', 'Taxonomy', 'Species', 'Subspecies', 'Conservation status'],
                    ['Description', 'Habitat', 'Diet', 'Behaviour', 'Breeding']
                ]
            else:
                animal_clusts = [
                    ['Distribution', 'Taxonomy', 'Species', 'Subspecies', 'Conservation status',
                     'Description', 'Habitat', 'Diet', 'Behaviour', 'Breeding']
                ]
        return animal_clusts

    elif args.category == 'company':
        if args.lead_section_num == 20:
            if args.topic_num == 4:
                # this is tailored for POWER
                # 20 titles
                company_clusts = [
                    ['Locations'],
                    ['Products', 'Services', 'Technology'],
                    ['Awards', 'Development'],
                    ['History', 'Ownership'],
                ]
            else:
                # 1 topic
                company_clusts = [
                    ['History', 'Products', 'Services', 'Destinations', 'Technology',
                     'Fleet', 'Subsidiaries', 'Operations', 'Locations',
                     'Awards']
                ]
        else:
            # 10 lead_section_num
            if last_noisy:
                company_clusts = [
                    ['History', 'Ownership'],
                    ['Products', 'Services', 'Destinations', 'Technology'],
                    ['Fleet', 'Subsidiaries', 'Operations', 'Locations'],
                    ['Awards'],
                    ['Status']
                ]
            else:
                company_clusts = [
                    ['History', ],
                    ['Products', 'Services', ],
                    ['Fleet', 'Subsidiaries', 'Locations'],
                    ['Awards']
                ]
        return company_clusts
    elif args.category == 'film':
        if args.lead_section_num == 32:
            if args.topic_num == 4:
                film_clusts = [
                    ['Cast', 'Casting', 'Roles', 'Starring'],
                    ['Plot', 'Summary', 'Story', 'Background'],
                    ['Career', 'Crew', 'Awards', 'Reception'],
                    ['Development', 'Music'],
                ]
        elif args.lead_section_num == 30:
            if args.topic_num == 4:
                film_clusts = [
                    ['Cast', 'Casting', 'Roles', 'Starring'],
                    ['Plot', 'Summary', 'Story'],
                    ['Reception', 'Critic', 'Awards', 'Nominations'],
                    ['Development', 'Box'],
                ]
        elif args.lead_section_num == 20:
            if args.topic_num == 4:
                # this is tailored for POWER
                # 20 titles
                film_clusts = [
                    ['Cast'],
                    ['Awards'],
                    ['Plot', 'Development'],
                    ['Production', 'Reception'],
                ]
            elif args.topic_num == 5:
                # this is tailored for POWER
                # 20 titles
                film_clusts = [
                    ['Cast', 'Casting', 'Roles', 'Starring'],
                    ['Plot', 'Summary', 'Story'],
                    ['Reception', 'Critic', 'Awards', 'Award', 'Nominations'],
                    ['Development', 'Box'],
                    ['Filming'],
                ]
            else:
                # 1 topic
                film_clusts = [
                    ['Cast', 'Casting', 'Plot', 'Synopsis',
                     'Reception', 'Awards', 'Accolades', 'Production', 'Filming', 'Development'],
                ]
        else:
            # 10 lead_section_num
            if last_noisy:
                film_clusts = [
                    ['Cast', 'Casting'],
                    ['Plot'],
                    ['Reception', 'Awards'],
                    ['Production'],
                    ['Development'],
                ]
            else:
                film_clusts = [
                    ['Cast', 'Casting'],
                    ['Plot'],
                    ['Reception', 'Awards'],
                    ['Production', 'Development'],
                ]
        return film_clusts
    animal_path = os.path.join(args.classifier_dir, 'animal', 'TopicList.txt')
    film_clusts = [
        ['Cast', 'Casting'],
        ['Plot', 'Synopsis', 'Plot summary'],
        ['Production', 'Filming', 'Development'],
        ['Reception', 'Critical reception', 'Critical response', 'Awards', 'Accolades', 'Awards and nominations'],
        ['Box office'],
    ]

    film_path = os.path.join(args.classifier_dir, 'film', 'TopicList.txt')
    with open(film_path, 'w') as fin:
        json.dump(film_clusts, fin, ensure_ascii=False)

    # 10 titles
    # company_clusts = [
    #     ['History'],
    #     ['Products', 'Services', 'Destinations'],
    #     ['Fleet', 'Operations'],
    #     ['Awards'],
    #     ['NOISE']
    # ]

    # 20 titles
    company_clusts = [
        ['History', 'Company history'],
        ['Products', 'Services', 'Destinations', 'Products and services', 'Technology'],
        ['Fleet', 'Subsidiaries', 'Operations', 'Locations'],
        ['Awards'],
    ]

    # 30 titles
    # company_clusts = [
    #      ['History', 'Company history', 'Ownership'],
    #      ['Products', 'Services', 'Destinations', 'Products and services', 'Technology'],
    #      ['Fleet', 'Subsidiaries', 'Operations', 'Locations'],
    #      ['Awards', 'Controversies', 'Controversy', 'Criticism', 'Accidents and incidents', 'Reception'],
    #      ['NOISE']
    #  ]

    company_path = os.path.join(args.classifier_dir, 'company', 'TopicList.txt')
    with open(company_path, 'w') as fin:
        json.dump(company_clusts, fin, ensure_ascii=False)


def get_inverse_pattern(args, title):
    assert args.prompt == 'inverse'
    patterns = [
        f" This document is about {title}."
    ]

    if args.category == 'animal':
        addition_patterns = [
            f" {title}'s distribution is mentioned in above sentences.",
            f" This document introduces subspecies of {title}.",
            f" This document describes {title}.",
            f" This document introduces conservation status of {title}."
        ]
    elif args.category == 'company':
        addition_patterns = []
    elif args.category == 'film':
        addition_patterns = []
    else:  # wcep
        patterns = []
        for wiki_link in title:
            entity = " ".join(wiki_link.split("/")[-1].split("_"))
            patterns.append(f" This document is about {entity}.")
        return patterns[:args.addition_pattern_num + 1]
    assert args.addition_pattern_num < len(addition_patterns)
    patterns.extend(addition_patterns[:args.addition_pattern_num])

    return patterns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal_topic_num', type=int, default=5)
    args = parser.parse_args()
